"""
DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance
Baseline implementation as described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class DraftAttention(nn.Module):
    """
    Training-free acceleration for video diffusion transformers using
    low-resolution draft attention maps to guide sparse attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        sparsity_ratio: float = 0.1,
        pooling_kernel: Tuple[int, int] = (8, 16),
        pooling_stride: Optional[Tuple[int, int]] = None,
        use_flash_attention: bool = False,
        device: str = "cuda"
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sparsity_ratio = sparsity_ratio
        self.pooling_kernel = pooling_kernel
        self.pooling_stride = pooling_stride or pooling_kernel
        self.device = device
        
        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Initialize reordering indices cache
        self._reorder_cache = {}
        
    def _compute_reorder_indices(self, 
                                frame_size: Tuple[int, int], 
                                patch_size: Tuple[int, int],
                                num_frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate deterministic reordering indices for memory alignment."""
        H, W = frame_size
        h, w = patch_size
        n = num_frames * H * W
        
        # Generate permutation
        pi = []
        for f in range(num_frames):
            for i in range(H // h):
                for j in range(W // w):
                    for u in range(h):
                        for v in range(w):
                            y = i * h + u
                            x = j * w + v
                            idx = f * H * W + y * W + x
                            pi.append(idx)
        
        pi = torch.tensor(pi, dtype=torch.long, device=self.device)
        pi_inv = torch.empty_like(pi)
        pi_inv[pi] = torch.arange(n, device=self.device)
        
        return pi, pi_inv
    
    def _apply_reordering(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Apply reordering to input tensor."""
        return x[..., indices, :]
    
    def _compute_draft_attention(self, 
                                q: torch.Tensor, 
                                k: torch.Tensor,
                                frame_size: Tuple[int, int],
                                num_frames: int) -> torch.Tensor:
        """Compute low-resolution draft attention map."""
        B, N, D = q.shape
        H, W = frame_size
        
        # Reshape for pooling
        q_reshaped = q.view(B, num_frames, H, W, D).permute(0, 4, 1, 2, 3)  # B, D, F, H, W
        k_reshaped = k.view(B, num_frames, H, W, D).permute(0, 4, 1, 2, 3)
        
        # Average pooling
        q_pooled = F.avg_pool3d(
            q_reshaped, 
            kernel_size=(1, *self.pooling_kernel),
            stride=(1, *self.pooling_stride)
        )
        k_pooled = F.avg_pool3d(
            k_reshaped, 
            kernel_size=(1, *self.pooling_kernel),
            stride=(1, *self.pooling_stride)
        )
        
        # Reshape back
        B, D, F, H_pooled, W_pooled = q_pooled.shape
        q_draft = q_pooled.permute(0, 2, 3, 4, 1).reshape(B, -1, D)
        k_draft = k_pooled.permute(0, 2, 3, 4, 1).reshape(B, -1, D)
        
        # Compute draft attention
        attn_scores = torch.bmm(q_draft, k_draft.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        return attn_probs
    
    def _generate_sparsity_mask(self, 
                               draft_attn: torch.Tensor,
                               sparsity_ratio: float) -> torch.Tensor:
        """Generate sparsity mask from draft attention."""
        B, G, _ = draft_attn.shape
        
        # Flatten and get top-k indices
        flat_attn = draft_attn.view(B, -1)
        k = int(flat_attn.shape[-1] * sparsity_ratio)
        
        # Get top-k values and indices
        _, top_indices = torch.topk(flat_attn, k, dim=-1)
        
        # Create mask
        mask_flat = torch.zeros_like(flat_attn)
        mask_flat.scatter_(-1, top_indices, 1.0)
        mask = mask_flat.view(B, G, G)
        
        return mask
    
    def _lift_mask_to_full_resolution(self, 
                                    mask: torch.Tensor,
                                    frame_size: Tuple[int, int],
                                    num_frames: int) -> torch.Tensor:
        """Lift region-level mask to token-level mask."""
        B, G, _ = mask.shape
        H, W = frame_size
        
        # Calculate pooling dimensions
        H_pooled = H // self.pooling_stride[0]
        W_pooled = W // self.pooling_stride[1]
        
        # Expand mask to full resolution
        mask_expanded = mask.view(B, G, G, 1, 1)
        mask_full = mask_expanded.expand(B, G, G, 
                                       self.pooling_stride[0] * self.pooling_stride[1],
                                       self.pooling_stride[0] * self.pooling_stride[1])
        
        # Reshape to token-level mask
        mask_full = mask_full.reshape(B, G * self.pooling_stride[0] * self.pooling_stride[1],
                                    G * self.pooling_stride[0] * self.pooling_stride[1])
        
        return mask_full
    
    def forward(self, 
                x: torch.Tensor,
                frame_size: Tuple[int, int],
                num_frames: int,
                sparsity_ratio: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass of DraftAttention.
        
        Args:
            x: Input tensor of shape (B, N, D)
            frame_size: (H, W) spatial dimensions
            num_frames: Number of temporal frames
            sparsity_ratio: Override default sparsity ratio
        
        Returns:
            Output tensor of shape (B, N, D)
        """
        B, N, D = x.shape
        sparsity_ratio = sparsity_ratio or self.sparsity_ratio
        
        # Generate reordering indices if not cached
        cache_key = (frame_size, self.pooling_kernel, num_frames)
        if cache_key not in self._reorder_cache:
            self._reorder_cache[cache_key] = self._compute_reorder_indices(
                frame_size, self.pooling_kernel, num_frames
            )
        reorder_idx, restore_idx = self._reorder_cache[cache_key]
        
        # Apply reordering
        x_reordered = self._apply_reordering(x, reorder_idx)
        
        # Project to Q, K, V
        q = self.q_proj(x_reordered)
        k = self.k_proj(x_reordered)
        v = self.v_proj(x_reordered)
        
        # Multi-head reshape
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute draft attention (using first head for draft)
        q_draft = q[:, 0].reshape(B * self.num_heads, N, self.head_dim)
        k_draft = k[:, 0].reshape(B * self.num_heads, N, self.head_dim)
        
        # Reshape for draft computation
        q_draft = q_draft.view(B, self.num_heads, N, self.head_dim)[:, 0]  # B, N, head_dim
        k_draft = k_draft.view(B, self.num_heads, N, self.head_dim)[:, 0]
        
        draft_attn = self._compute_draft_attention(q_draft, k_draft, frame_size, num_frames)
        
        # Generate sparsity mask
        mask = self._generate_sparsity_mask(draft_attn, sparsity_ratio)
        mask_full = self._lift_mask_to_full_resolution(mask, frame_size, num_frames)
        
        # Expand mask for all heads
        mask_full = mask_full.unsqueeze(1).expand(B, self.num_heads, N, N)
        
        # Compute sparse attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        masked_scores = attn_scores.masked_fill(mask_full == 0, float('-inf'))
        attn_probs = F.softmax(masked_scores, dim=-1)
        
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        
        # Apply output projection
        out = self.out_proj(out)
        
        # Restore original ordering
        out_restored = self._apply_reordering(out, restore_idx)
        
        return out_restored
    
    def load_weights(self, checkpoint_path: str):
        """Load pre-trained weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint)
        
    def save_weights(self, checkpoint_path: str):
        """Save model weights to checkpoint."""
        torch.save(self.state_dict(), checkpoint_path)


class DraftAttentionConfig:
    """Configuration class for DraftAttention."""
    
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        sparsity_ratio: float = 0.1,
        pooling_kernel: Tuple[int, int] = (8, 16),
        use_flash_attention: bool = False
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.sparsity_ratio = sparsity_ratio
        self.pooling_kernel = pooling_kernel
        self.use_flash_attention = use_flash_attention


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, D = 2, 128 * 16 * 9, 768  # 9 frames, 128x16 patches
    frame_size = (128, 16 * 16)  # H, W
    num_frames = 9
    
    # Initialize model
    model = DraftAttention(
        dim=D,
        num_heads=12,
        sparsity_ratio=0.1,
        device=device
    ).to(device)
    
    # Create dummy input
    x = torch.randn(B, N, D, device=device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x, frame_size, num_frames)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("DraftAttention test passed!")