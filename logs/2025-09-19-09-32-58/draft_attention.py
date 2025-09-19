"""
DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance
Implementation of the training-free framework for accelerating video diffusion transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class DraftAttention(nn.Module):
    """
    Training-free sparse attention mechanism for video diffusion transformers.
    
    This module implements the DraftAttention method which uses low-resolution
    attention maps to guide sparse attention computation in full resolution.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        sparsity_ratio: float = 0.8,
        pooling_kernel: Tuple[int, int] = (8, 16),
        **kwargs
    ):
        """
        Initialize DraftAttention module.
        
        Args:
            dim: Hidden dimension
            num_heads: Number of attention heads
            sparsity_ratio: Target sparsity ratio (0.5-0.9)
            pooling_kernel: Kernel size for average pooling (height, width)
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sparsity_ratio = sparsity_ratio
        self.pooling_kernel = pooling_kernel
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Calculate reduction factor based on pooling kernel
        self.reduction_factor = pooling_kernel[0] * pooling_kernel[1]
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Scale factor for attention
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def _create_draft_attention_map(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor
    ) -> torch.Tensor:
        """
        Create low-resolution draft attention map using average pooling.
        
        Args:
            q: Query tensor [B, H, N, D]
            k: Key tensor [B, H, N, D]
            
        Returns:
            Draft attention map [B, H, N', N']
        """
        B, H, N, D = q.shape
        
        # Calculate spatial dimensions based on pooling kernel
        # Assuming tokens are arranged in (T, H, W) format
        sqrt_n = int(math.sqrt(N))
        if sqrt_n * sqrt_n == N:
            # Square spatial layout
            h = w = sqrt_n
            t = 1
        else:
            # Try rectangular layout (common for video: T×H×W)
            # For video: typically T×H×W tokens
            # This is a heuristic - adjust based on actual use case
            t = max(1, N // (self.pooling_kernel[1] * 16))  # Assume 16:9 aspect
            hw = N // t
            h = int(math.sqrt(hw * 9 / 16))  # 16:9 aspect ratio
            w = hw // h if h > 0 else 1
            
        # Reshape for pooling: [B*H, D, T, H, W]
        q_reshaped = q.transpose(1, 2).reshape(B * H, D, t, h, w)
        k_reshaped = k.transpose(1, 2).reshape(B * H, D, t, h, w)
        
        # Apply average pooling
        pool_t = max(1, t // self.pooling_kernel[0])
        pool_h = max(1, h // self.pooling_kernel[0])
        pool_w = max(1, w // self.pooling_kernel[1])
        
        q_pooled = F.adaptive_avg_pool3d(q_reshaped, (pool_t, pool_h, pool_w))
        k_pooled = F.adaptive_avg_pool3d(k_reshaped, (pool_t, pool_h, pool_w))
        
        # Reshape back for attention computation
        q_pooled = q_pooled.reshape(B, H, D, -1).transpose(2, 3)
        k_pooled = k_pooled.reshape(B, H, D, -1).transpose(2, 3)
        
        # Compute draft attention
        draft_attn = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) * self.scale
        draft_attn = F.softmax(draft_attn, dim=-1)
        
        return draft_attn
    
    def _create_sparsity_mask(
        self,
        draft_attn: torch.Tensor,
        sparsity_ratio: Optional[float] = None
    ) -> torch.Tensor:
        """
        Create sparsity mask based on draft attention map.
        
        Args:
            draft_attn: Draft attention map [B, H, N', N']
            sparsity_ratio: Target sparsity ratio
            
        Returns:
            Binary mask indicating which tokens to keep
        """
        if sparsity_ratio is None:
            sparsity_ratio = self.sparsity_ratio
            
        B, H, N_draft, _ = draft_attn.shape
        
        # Calculate number of tokens to keep
        keep_ratio = 1.0 - sparsity_ratio
        num_keep = max(1, int(N_draft * keep_ratio))
        
        # Find top-k attention values for each query
        # Use max pooling across key dimension to get importance scores
        importance = draft_attn.max(dim=-1)[0]  # [B, H, N_draft]
        
        # Get indices of top-k important regions
        _, top_indices = torch.topk(importance, num_keep, dim=-1, sorted=False)
        
        # Create binary mask
        mask = torch.zeros_like(importance, dtype=torch.bool)
        mask.scatter_(-1, top_indices, True)
        
        return mask
    
    def _reorder_tokens(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reorder tokens based on sparsity mask for efficient computation.
        
        Args:
            x: Input tensor [B, N, D]
            mask: Binary mask [B, N']
            
        Returns:
            Reordered tensor and indices for restoration
        """
        B, N, D = x.shape
        
        # Expand mask to full resolution
        mask_expanded = mask.repeat_interleave(self.reduction_factor, dim=-1)
        mask_expanded = mask_expanded[:, :N]  # Truncate if needed
        
        # Get indices of selected tokens
        indices = torch.where(mask_expanded)[1].reshape(B, -1)
        
        # Reorder tokens
        batch_indices = torch.arange(B, device=x.device).unsqueeze(-1)
        x_reordered = x[batch_indices, indices]
        
        return x_reordered, indices
    
    def _restore_order(
        self,
        x_sparse: torch.Tensor,
        indices: torch.Tensor,
        original_shape: torch.Size
    ) -> torch.Tensor:
        """
        Restore original token order after sparse attention.
        
        Args:
            x_sparse: Output from sparse attention
            indices: Indices used for reordering
            original_shape: Original tensor shape
            
        Returns:
            Tensor with restored order
        """
        B, N, D = original_shape
        device = x_sparse.device
        
        # Initialize output tensor
        x_restored = torch.zeros(original_shape, device=device, dtype=x_sparse.dtype)
        
        # Scatter values back to original positions
        batch_indices = torch.arange(B, device=device).unsqueeze(-1)
        x_restored[batch_indices, indices] = x_sparse
        
        return x_restored
    
    def forward(
        self,
        x: torch.Tensor,
        sparsity_ratio: Optional[float] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of DraftAttention.
        
        Args:
            x: Input tensor [B, N, D]
            sparsity_ratio: Override default sparsity ratio
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor [B, N, D]
        """
        B, N, D = x.shape
        H = self.num_heads
        
        # Project to Q, K, V
        q = self.q_proj(x).reshape(B, N, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, H, self.head_dim).transpose(1, 2)
        
        # Create draft attention map
        draft_attn = self._create_draft_attention_map(q, k)
        
        # Create sparsity mask
        mask = self._create_sparsity_mask(draft_attn, sparsity_ratio)
        
        # Reorder tokens for sparse computation
        q_flat = q.transpose(1, 2).reshape(B, N, D)
        k_flat = k.transpose(1, 2).reshape(B, N, D)
        v_flat = v.transpose(1, 2).reshape(B, N, D)
        
        q_reordered, q_indices = self._reorder_tokens(q_flat, mask)
        k_reordered, k_indices = self._reorder_tokens(k_flat, mask)
        v_reordered, v_indices = self._reorder_tokens(v_flat, mask)
        
        # Reshape back for multi-head attention
        N_sparse = q_reordered.shape[1]
        q_sparse = q_reordered.reshape(B, N_sparse, H, self.head_dim).transpose(1, 2)
        k_sparse = k_reordered.reshape(B, N_sparse, H, self.head_dim).transpose(1, 2)
        v_sparse = v_reordered.reshape(B, N_sparse, H, self.head_dim).transpose(1, 2)
        
        # Compute sparse attention
        attn_weights = torch.matmul(q_sparse, k_sparse.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out_sparse = torch.matmul(attn_weights, v_sparse)
        
        # Restore original order
        out_flat = out_sparse.transpose(1, 2).reshape(B, N_sparse, D)
        out_restored = self._restore_order(out_flat, q_indices, (B, N, D))
        
        # Final projection
        output = self.out_proj(out_restored)
        
        if return_attention:
            return output, attn_weights
        
        return output
    
    def load_weights(self, state_dict: dict):
        """Load pre-trained weights into the module."""
        self.load_state_dict(state_dict)


class DraftAttentionBlock(nn.Module):
    """
    Transformer block using DraftAttention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        sparsity_ratio: float = 0.8,
        **kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DraftAttention(dim, num_heads, sparsity_ratio, **kwargs)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transformer block."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def create_draft_attention_model(
    dim: int = 768,
    num_blocks: int = 12,
    num_heads: int = 12,
    sparsity_ratio: float = 0.8,
    **kwargs
) -> nn.Module:
    """
    Create a transformer model with DraftAttention blocks.
    
    Args:
        dim: Hidden dimension
        num_blocks: Number of transformer blocks
        num_heads: Number of attention heads
        sparsity_ratio: Global sparsity ratio
        
    Returns:
        Transformer model with DraftAttention
    """
    blocks = []
    for i in range(num_blocks):
        # Use different sparsity ratios for different layers
        layer_sparsity = max(0.5, sparsity_ratio - 0.1 * (i % 3))
        blocks.append(DraftAttentionBlock(
            dim, num_heads, sparsity_ratio=layer_sparsity, **kwargs
        ))
    
    return nn.Sequential(*blocks)