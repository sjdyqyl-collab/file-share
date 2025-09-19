import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import math


class DraftAttention(nn.Module):
    """
    DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance
    
    This is the original implementation as described in the paper.
    Uses low-resolution draft attention to guide sparse full-resolution attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        sparsity_ratio: float = 0.75,
        pooling_kernel: Tuple[int, int] = (8, 16),
        pooling_stride: Optional[Tuple[int, int]] = None,
        use_flash_attention: bool = False,
    ):
        """
        Initialize DraftAttention module.
        
        Args:
            dim: Hidden dimension size
            num_heads: Number of attention heads
            sparsity_ratio: Sparsity ratio for attention (0.55, 0.75, 0.9)
            pooling_kernel: Kernel size for average pooling (height, width)
            pooling_stride: Stride for pooling (defaults to kernel size)
            use_flash_attention: Whether to use FlashAttention-style kernels
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sparsity_ratio = sparsity_ratio
        
        self.pooling_kernel = pooling_kernel
        self.pooling_stride = pooling_stride or pooling_kernel
        
        self.use_flash_attention = use_flash_attention
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Scale factor for attention
        self.scale = self.head_dim ** -0.5
        
    def _compute_reorder_indices(self, 
                                frame_size: Tuple[int, int], 
                                patch_size: Tuple[int, int],
                                num_frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate reorder and restore indices for hardware-friendly execution.
        
        Args:
            frame_size: (H, W) frame dimensions
            patch_size: (h, w) patch dimensions for grouping
            num_frames: Number of frames
            
        Returns:
            reorder_idx: Tensor of indices for reordering
            restore_idx: Tensor of indices for restoring original order
        """
        H, W = frame_size
        h, w = patch_size
        
        # Ensure dimensions are divisible
        assert H % h == 0 and W % w == 0, "Frame dimensions must be divisible by patch size"
        
        reorder_idx = []
        
        for f in range(num_frames):
            for i in range(H // h):
                for j in range(W // w):
                    for u in range(h):
                        for v in range(w):
                            y = i * h + u
                            x = j * w + v
                            idx = f * H * W + y * W + x
                            reorder_idx.append(idx)
        
        reorder_idx = torch.tensor(reorder_idx, dtype=torch.long)
        
        # Create restore indices (inverse permutation)
        restore_idx = torch.empty_like(reorder_idx)
        restore_idx[reorder_idx] = torch.arange(len(reorder_idx))
        
        return reorder_idx, restore_idx
    
    def _average_pool_features(self, x: torch.Tensor, 
                             original_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Apply average pooling to create low-resolution draft features.
        
        Args:
            x: Input tensor [B, N, C] where N = F*H*W
            original_shape: (B, F, H, W, C) original shape
            
        Returns:
            Pooled features [B, G, C] where G = F*(H//kh)*(W//kw)
        """
        B, N, C = x.shape
        B, F, H, W, _ = original_shape
        
        # Reshape to spatial format
        x_reshaped = x.view(B, F, H, W, C).permute(0, 1, 4, 2, 3)  # [B, F, C, H, W]
        
        # Apply average pooling
        pooled = F.avg_pool2d(
            x_reshaped.view(B * F, C, H, W),
            kernel_size=self.pooling_kernel,
            stride=self.pooling_stride
        )
        
        # Reshape back
        _, C_pooled, H_pooled, W_pooled = pooled.shape
        pooled = pooled.view(B, F, C_pooled, H_pooled, W_pooled)
        pooled = pooled.permute(0, 1, 3, 4, 2).contiguous()  # [B, F, H_p, W_p, C]
        
        # Flatten spatial dimensions
        pooled = pooled.view(B, -1, C_pooled)  # [B, G, C]
        
        return pooled
    
    def _compute_draft_attention(self, q_draft: torch.Tensor, 
                               k_draft: torch.Tensor) -> torch.Tensor:
        """
        Compute low-resolution draft attention map.
        
        Args:
            q_draft: Query tensor [B, G, C]
            k_draft: Key tensor [B, G, C]
            
        Returns:
            Draft attention map [B, G, G]
        """
        B, G, C = q_draft.shape
        
        # Reshape for multi-head attention
        q_draft = q_draft.view(B, G, self.num_heads, self.head_dim).transpose(1, 2)
        k_draft = k_draft.view(B, G, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q_draft, k_draft.transpose(-2, -1)) * self.scale
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)  # [B, num_heads, G, G]
        
        # Average across heads for guidance
        attn_weights = attn_weights.mean(dim=1)  # [B, G, G]
        
        return attn_weights
    
    def _create_sparsity_mask(self, draft_attn: torch.Tensor, 
                            original_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Create sparsity mask based on draft attention.
        
        Args:
            draft_attn: Draft attention map [B, G, G]
            original_shape: (B, F, H, W, C) original shape
            
        Returns:
            Sparsity mask [B, N, N] where N = F*H*W
        """
        B, G, G = draft_attn.shape
        B, F, H, W, C = original_shape
        
        # Compute number of tokens to keep
        kh, kw = self.pooling_kernel
        G_h = H // kh
        G_w = W // kw
        
        # Flatten draft attention
        draft_attn_flat = draft_attn.view(B, -1)  # [B, G*G]
        
        # Determine threshold for top-k selection
        k = int(G * G * self.sparsity_ratio)
        
        # Get top-k indices
        _, top_indices = torch.topk(draft_attn_flat, k, dim=1)
        
        # Create binary mask at region level
        region_mask = torch.zeros(B, G, G, device=draft_attn.device)
        batch_indices = torch.arange(B).unsqueeze(1).expand(-1, k)
        flat_indices = top_indices
        
        # Convert flat indices to 2D indices
        row_indices = flat_indices // G
        col_indices = flat_indices % G
        
        region_mask[batch_indices, row_indices, col_indices] = 1.0
        
        # Expand mask to full resolution
        N = F * H * W
        full_mask = torch.zeros(B, N, N, device=draft_attn.device)
        
        # Map region-level mask to token-level mask
        for b in range(B):
            for g_i in range(G):
                for g_j in range(G):
                    if region_mask[b, g_i, g_j] > 0:
                        # Map region indices to token indices
                        region_i_start = g_i * kh * kw
                        region_i_end = (g_i + 1) * kh * kw
                        region_j_start = g_j * kh * kw
                        region_j_end = (g_j + 1) * kh * kw
                        
                        # Handle frame boundaries
                        for f in range(F):
                            token_i_start = f * H * W + (g_i // G_w) * kh * W + (g_i % G_w) * kw
                            token_i_end = token_i_start + kh * kw
                            token_j_start = f * H * W + (g_j // G_w) * kh * W + (g_j % G_w) * kw
                            token_j_end = token_j_start + kh * kw
                            
                            full_mask[b, token_i_start:token_i_end, token_j_start:token_j_end] = 1.0
        
        return full_mask
    
    def forward(
        self, 
        x: torch.Tensor,
        frame_size: Optional[Tuple[int, int]] = None,
        num_frames: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass of DraftAttention.
        
        Args:
            x: Input tensor [B, N, C] where N = F*H*W
            frame_size: (H, W) frame dimensions. If None, inferred from x
            num_frames: Number of frames. If None, inferred from x
            
        Returns:
            Output tensor [B, N, C]
        """
        B, N, C = x.shape
        
        # Infer frame_size and num_frames if not provided
        if frame_size is None or num_frames is None:
            # Assume square frames for inference
            sqrt_n = int(math.sqrt(N))
            if sqrt_n * sqrt_n == N:
                frame_size = (sqrt_n, sqrt_n)
                num_frames = 1
            else:
                # Common video diffusion sizes
                if N == 128 * 32 * 48:  # Wan2.1 512p
                    frame_size = (32, 48)
                    num_frames = 128
                elif N == 128 * 48 * 80:  # HunyuanVideo 768p
                    frame_size = (48, 80)
                    num_frames = 128
                else:
                    raise ValueError(f"Cannot infer frame_size and num_frames from N={N}")
        
        H, W = frame_size
        F = num_frames
        
        # Ensure dimensions are compatible
        assert N == F * H * W, f"N={N} must equal F*H*W={F}*{H}*{W}={F*H*W}"
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Create draft features via average pooling
        original_shape = (B, F, H, W, C)
        q_draft = self._average_pool_features(q, original_shape)
        k_draft = self._average_pool_features(k, original_shape)
        
        # Compute draft attention
        draft_attn = self._compute_draft_attention(q_draft, k_draft)
        
        # Create sparsity mask
        sparsity_mask = self._create_sparsity_mask(draft_attn, original_shape)
        
        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute full attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply sparsity mask
        scores = scores * sparsity_mask.unsqueeze(1)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)
        
        return out
    
    def load_weights(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """Load weights from state dict."""
        self.load_state_dict(state_dict, strict=strict)
    
    def get_sparsity_stats(self) -> Dict[str, float]:
        """Get sparsity statistics."""
        return {
            'sparsity_ratio': self.sparsity_ratio,
            'pooling_kernel': self.pooling_kernel,
            'pooling_stride': self.pooling_stride
        }