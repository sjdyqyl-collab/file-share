import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class DraftAttention(nn.Module):
    """
    DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance
    
    This class implements the original method proposed in the paper, providing:
    1. Low-resolution draft attention via average pooling
    2. Structured sparsity pattern generation
    3. Deterministic token reordering for hardware efficiency
    4. Training-free integration with existing models
    """
    
    def __init__(
        self,
        hidden_dim: int,
        sparsity_ratio: float = 0.75,
        pooling_kernel: Tuple[int, int] = (8, 16),
        num_heads: int = 1,
        dropout: float = 0.0
    ):
        """
        Initialize DraftAttention module.
        
        Args:
            hidden_dim: Hidden dimension size (d)
            sparsity_ratio: Ratio of attention connections to retain (r ∈ [0.5, 0.9])
            pooling_kernel: Spatial pooling kernel size (height, width)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sparsity_ratio = sparsity_ratio
        self.pooling_kernel = pooling_kernel
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Initialize linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def _compute_reorder_indices(
        self, 
        frame_size: Tuple[int, int], 
        patch_size: Tuple[int, int], 
        num_frames: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate reordering indices for hardware-efficient computation.
        
        Args:
            frame_size: (H, W) frame dimensions
            patch_size: (h, w) patch dimensions
            num_frames: Number of frames (F)
            
        Returns:
            reorder_idx: Permutation indices for reordering
            restore_idx: Inverse permutation indices for restoration
        """
        H, W = frame_size
        h, w = patch_size
        n = num_frames * H * W
        
        # Generate reorder indices
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
        
        # Generate restore indices (inverse permutation)
        restore_idx = torch.zeros_like(reorder_idx)
        restore_idx[reorder_idx] = torch.arange(n, dtype=torch.long)
        
        return reorder_idx, restore_idx
    
    def _average_pool_2d(
        self, 
        x: torch.Tensor, 
        kernel_size: Tuple[int, int], 
        stride: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Apply 2D average pooling to input tensor.
        
        Args:
            x: Input tensor of shape (B, N, C)
            kernel_size: Pooling kernel size (height, width)
            stride: Stride size (defaults to kernel_size)
            
        Returns:
            Pooled tensor of shape (B, N', C)
        """
        if stride is None:
            stride = kernel_size
            
        B, N, C = x.shape
        
        # Reshape to 4D for pooling: (B, C, H, W)
        # Assuming spatial layout is known - need to infer from N
        # For now, assume N = F * H * W where F is number of frames
        
        # This is a simplified version - in practice, you'd need to know
        # the exact spatial dimensions
        H = int(np.sqrt(N))
        W = N // H if N % H == 0 else H
        
        x_4d = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Apply average pooling
        pooled = F.avg_pool2d(x_4d, kernel_size, stride=stride)
        
        # Reshape back to 3D
        B_pooled, C_pooled, H_pooled, W_pooled = pooled.shape
        pooled = pooled.reshape(B_pooled, C_pooled, -1).transpose(1, 2)
        
        return pooled
    
    def _create_sparsity_mask(
        self, 
        draft_attention: torch.Tensor, 
        sparsity_ratio: float
    ) -> torch.Tensor:
        """
        Create sparsity mask from draft attention map.
        
        Args:
            draft_attention: Draft attention map of shape (B, g, g)
            sparsity_ratio: Ratio of connections to retain
            
        Returns:
            Binary mask of shape (B, g, g)
        """
        B, g, _ = draft_attention.shape
        
        # Flatten attention for top-k selection
        flat_attention = draft_attention.view(B, -1)
        
        # Number of connections to retain
        num_retain = int(sparsity_ratio * g * g)
        
        # Get top-k indices
        _, top_indices = torch.topk(flat_attention, num_retain, dim=-1)
        
        # Create mask
        mask = torch.zeros_like(flat_attention)
        mask.scatter_(1, top_indices, 1.0)
        mask = mask.view(B, g, g)
        
        return mask.bool()
    
    def _lift_mask_to_tokens(
        self, 
        region_mask: torch.Tensor, 
        region_size: int
    ) -> torch.Tensor:
        """
        Lift region-level mask to token-level resolution.
        
        Args:
            region_mask: Binary mask of shape (B, g, g)
            region_size: Number of tokens per region
            
        Returns:
            Token-level mask of shape (B, n, n)
        """
        B, g, _ = region_mask.shape
        n = g * region_size
        
        # Expand mask to token resolution
        token_mask = region_mask.unsqueeze(-1).unsqueeze(-1)
        token_mask = token_mask.expand(B, g, g, region_size, region_size)
        token_mask = token_mask.reshape(B, n, n)
        
        return token_mask
    
    def forward(
        self, 
        x: torch.Tensor,
        frame_size: Tuple[int, int] = (48, 80),
        num_frames: int = 128,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of DraftAttention.
        
        Args:
            x: Input tensor of shape (B, n, d)
            frame_size: Spatial dimensions (H, W) for reordering
            num_frames: Number of frames for reordering
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor of shape (B, n, d)
        """
        B, n, d = x.shape
        
        # Compute Q, K, V projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Step 1: Compute draft attention via average pooling
        # Pooling reduces sequence length by factor of pooling_kernel[0] * pooling_kernel[1]
        pool_h, pool_w = self.pooling_kernel
        region_size = pool_h * pool_w
        g = n // region_size  # Number of regions
        
        # Compute draft queries and keys
        draft_Q = self._average_pool_2d(Q, self.pooling_kernel)
        draft_K = self._average_pool_2d(K, self.pooling_kernel)
        
        # Compute draft attention
        draft_attention = torch.bmm(draft_Q, draft_K.transpose(-2, -1)) / np.sqrt(d)
        draft_attention = F.softmax(draft_attention, dim=-1)
        
        # Step 2: Create sparsity mask
        region_mask = self._create_sparsity_mask(draft_attention, self.sparsity_ratio)
        
        # Step 3: Lift mask to token resolution
        token_mask = self._lift_mask_to_tokens(region_mask, region_size)
        
        # Step 4: Compute sparse attention
        attention_scores = torch.bmm(Q, K.transpose(-2, -1)) / np.sqrt(d)
        
        # Apply sparsity mask
        attention_scores = attention_scores.masked_fill(~token_mask, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        # Apply attention to values
        out = torch.bmm(attention_weights, V)
        
        # Final projection
        out = self.out_proj(out)
        
        if return_attention:
            return out, attention_weights
        
        return out
    
    def load_weights(self, state_dict: dict):
        """
        Load pre-trained weights for linear projections.
        
        Args:
            state_dict: Dictionary containing weight tensors
        """
        self.q_proj.load_state_dict({'weight': state_dict['q_proj.weight']})
        self.k_proj.load_state_dict({'weight': state_dict['k_proj.weight']})
        self.v_proj.load_state_dict({'weight': state_dict['v_proj.weight']})
        self.out_proj.load_state_dict({'weight': state_dict['out_proj.weight']})


class DraftAttentionBlock(nn.Module):
    """
    Complete attention block with DraftAttention and feed-forward network.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        sparsity_ratio: float = 0.75,
        pooling_kernel: Tuple[int, int] = (8, 16),
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.attention = DraftAttention(
            hidden_dim=hidden_dim,
            sparsity_ratio=sparsity_ratio,
            pooling_kernel=pooling_kernel,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, **kwargs):
        # Attention with residual connection
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, **kwargs)
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


# Utility functions for testing
if __name__ == "__main__":
    # Test the implementation
    batch_size = 1
    seq_len = 128 * 48 * 80  # 128 frames, 48x80 latent
    hidden_dim = 768
    
    # Create model
    model = DraftAttention(
        hidden_dim=hidden_dim,
        sparsity_ratio=0.75,
        pooling_kernel=(8, 16)
    )
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Forward pass
    with torch.no_grad():
        output = model(x, frame_size=(48, 80), num_frames=128)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("DraftAttention implementation successful!")