"""
DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance
Implementation of the core method proposed in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class DraftAttention(nn.Module):
    """
    DraftAttention: Training-free acceleration for video diffusion transformers
    using low-resolution draft attention maps to guide sparse attention computation.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        sparsity_ratio: float = 0.75,
        pooling_kernel: Tuple[int, int] = (8, 16),
        use_full_attention_steps: float = 0.25,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            dim: Hidden dimension size
            num_heads: Number of attention heads
            sparsity_ratio: Fraction of tokens to retain in sparse attention (0-1)
            pooling_kernel: Kernel size for downsampling (height, width)
            use_full_attention_steps: Fraction of denoising steps to use full attention
            device: Device to run computations on
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sparsity_ratio = sparsity_ratio
        self.pooling_kernel = pooling_kernel
        self.use_full_attention_steps = use_full_attention_steps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
    
    def _compute_draft_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        height: int, 
        width: int
    ) -> torch.Tensor:
        """
        Compute low-resolution draft attention map.
        
        Args:
            q: Query tensor [B, H, N, D]
            k: Key tensor [B, H, N, D]
            height: Height of spatial feature map
            width: Width of spatial feature map
            
        Returns:
            Draft attention map [B, H, g_h, g_w, g_h, g_w]
        """
        B, H, N, D = q.shape
        
        # Reshape to spatial format [B, H, H_sp, W_sp, D]
        q_spatial = q.reshape(B, H, height, width, D)
        k_spatial = k.reshape(B, H, height, width, D)
        
        # Downsample using average pooling
        # Note: pooling_kernel is (h, w) for spatial dimensions
        q_draft = F.avg_pool2d(
            q_spatial.permute(0, 1, 4, 2, 3).reshape(B * H, D, height, width),
            kernel_size=self.pooling_kernel,
            stride=self.pooling_kernel
        ).reshape(B, H, D, -1).permute(0, 1, 3, 2)  # [B, H, g, D]
        
        k_draft = F.avg_pool2d(
            k_spatial.permute(0, 1, 4, 2, 3).reshape(B * H, D, height, width),
            kernel_size=self.pooling_kernel,
            stride=self.pooling_kernel
        ).reshape(B, H, D, -1).permute(0, 1, 3, 2)  # [B, H, g, D]
        
        # Compute draft attention
        draft_attention = torch.matmul(q_draft, k_draft.transpose(-2, -1)) / math.sqrt(D)
        draft_attention = F.softmax(draft_attention, dim=-1)
        
        return draft_attention
    
    def _create_sparsity_mask(
        self, 
        draft_attention: torch.Tensor, 
        height: int, 
        width: int
    ) -> torch.Tensor:
        """
        Create sparsity mask based on draft attention map.
        
        Args:
            draft_attention: Draft attention map [B, H, g, g]
            height: Height of spatial feature map
            width: Width of spatial feature map
            
        Returns:
            Binary sparsity mask [B, 1, N, N]
        """
        B, H, g, _ = draft_attention.shape
        
        # Average across heads for consistent mask
        draft_mean = draft_attention.mean(dim=1)  # [B, g, g]
        
        # Determine number of regions to keep
        num_keep = max(1, int(g * g * self.sparsity_ratio))
        
        # Find top-k values and create mask
        flat_attention = draft_mean.reshape(B, -1)
        _, top_indices = torch.topk(flat_attention, num_keep, dim=-1)
        
        # Create region-level mask
        region_mask = torch.zeros_like(flat_attention)
        region_mask.scatter_(1, top_indices, 1.0)
        region_mask = region_mask.reshape(B, g, g)
        
        # Expand to full resolution
        h_patches = height // self.pooling_kernel[0]
        w_patches = width // self.pooling_kernel[1]
        
        # Create full resolution mask
        full_mask = region_mask.repeat_interleave(self.pooling_kernel[0], dim=1)
        full_mask = full_mask.repeat_interleave(self.pooling_kernel[1], dim=2)
        
        # Ensure correct dimensions
        expected_height = h_patches * self.pooling_kernel[0]
        expected_width = w_patches * self.pooling_kernel[1]
        
        if full_mask.shape[1] != expected_height or full_mask.shape[2] != expected_width:
            full_mask = full_mask[:, :expected_height, :expected_width]
        
        # Flatten to token sequence
        full_mask = full_mask.reshape(B, -1, 1)
        full_mask = full_mask @ full_mask.transpose(-2, -1)  # [B, N, N]
        
        return full_mask.unsqueeze(1)  # [B, 1, N, N]
    
    def _reorder_tokens(
        self, 
        x: torch.Tensor, 
        height: int, 
        width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reorder tokens for hardware-friendly access.
        
        Args:
            x: Input tensor [B, N, D]
            height: Height of spatial feature map
            width: Width of spatial feature map
            
        Returns:
            Reordered tensor and permutation indices
        """
        B, N, D = x.shape
        
        # Reshape to spatial format
        x_spatial = x.reshape(B, height, width, D)
        
        # Create patches and reorder
        patch_h = self.pooling_kernel[0]
        patch_w = self.pooling_kernel[1]
        
        # Pad if necessary to make divisible
        pad_h = (patch_h - height % patch_h) % patch_h
        pad_w = (patch_w - width % patch_w) % patch_w
        
        if pad_h > 0 or pad_w > 0:
            x_spatial = F.pad(x_spatial, (0, 0, 0, pad_w, 0, pad_h))
            new_height = height + pad_h
            new_width = width + pad_w
        else:
            new_height = height
            new_width = width
        
        # Reshape into patches and flatten
        x_patches = x_spatial.reshape(
            B, 
            new_height // patch_h, patch_h,
            new_width // patch_w, patch_w,
            D
        )
        x_reordered = x_patches.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_reordered = x_reordered.reshape(B, -1, D)
        
        # Create permutation indices for restoration
        perm_indices = self._create_permutation_indices(new_height, new_width, patch_h, patch_w)
        
        return x_reordered, perm_indices
    
    def _create_permutation_indices(
        self, 
        height: int, 
        width: int, 
        patch_h: int, 
        patch_w: int
    ) -> torch.Tensor:
        """Create permutation indices for restoring original order."""
        indices = torch.arange(height * width, device=self.device).reshape(height, width)
        
        # Reshape into patches
        indices_patches = indices.reshape(
            height // patch_h, patch_h,
            width // patch_w, patch_w
        )
        indices_reordered = indices_patches.permute(0, 2, 1, 3).contiguous()
        
        return indices_reordered.reshape(-1)
    
    def _restore_order(self, x: torch.Tensor, perm_indices: torch.Tensor) -> torch.Tensor:
        """Restore original token order."""
        B, N, D = x.shape
        
        # Create inverse permutation
        inv_perm = torch.empty_like(perm_indices)
        inv_perm[perm_indices] = torch.arange(N, device=perm_indices.device)
        
        return x[:, inv_perm, :]
    
    def forward(
        self, 
        x: torch.Tensor,
        height: int,
        width: int,
        step_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Forward pass of DraftAttention.
        
        Args:
            x: Input tensor [B, N, D]
            height: Height of spatial feature map
            width: Width of spatial feature map
            step_ratio: Current denoising step ratio (0-1)
            
        Returns:
            Output tensor [B, N, D]
        """
        B, N, D = x.shape
        
        # Use full attention for early steps
        if step_ratio < self.use_full_attention_steps:
            return self._full_attention(x)
        
        # Compute Q, K, V
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute draft attention
        draft_attention = self._compute_draft_attention(q, k, height, width)
        
        # Create sparsity mask
        sparsity_mask = self._create_sparsity_mask(draft_attention, height, width)
        
        # Reorder tokens for hardware efficiency
        q_reordered, perm_indices = self._reorder_tokens(q.transpose(1, 2).reshape(B, N, D), height, width)
        q_reordered = q_reordered.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_reordered, _ = self._reorder_tokens(k.transpose(1, 2).reshape(B, N, D), height, width)
        k_reordered = k_reordered.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        v_reordered, _ = self._reorder_tokens(v.transpose(1, 2).reshape(B, N, D), height, width)
        v_reordered = v_reordered.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply sparsity mask and compute attention
        # For simplicity, we'll use masked attention
        attention_scores = torch.matmul(q_reordered, k_reordered.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Expand mask for all heads
        mask = sparsity_mask.expand(-1, self.num_heads, -1, -1)
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights.masked_fill(mask == 0, 0.0)  # Ensure exact sparsity
        
        # Apply attention to values
        out = torch.matmul(attention_weights, v_reordered)
        
        # Restore original order
        out_flat = out.transpose(1, 2).reshape(B, -1, self.num_heads * self.head_dim)
        out_restored = self._restore_order(out_flat, perm_indices)
        
        # Handle padding
        if out_restored.shape[1] > N:
            out_restored = out_restored[:, :N, :]
        
        # Final projection
        return self.out_proj(out_restored)
    
    def _full_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Standard full attention computation."""
        B, N, D = x.shape
        
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, v)
        
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)
    
    def load_weights(self, state_dict: dict):
        """Load pre-trained weights."""
        self.load_state_dict(state_dict)
    
    def save_weights(self, path: str):
        """Save model weights."""
        torch.save(self.state_dict(), path)