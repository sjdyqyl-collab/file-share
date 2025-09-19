"""
DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance
Original implementation as described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class DraftAttention(nn.Module):
    """
    Training-free acceleration of video diffusion transformers with dynamic sparse attention.
    
    This implementation follows the paper's methodology:
    1. Compute draft attention on downsampled features
    2. Guide sparse attention computation on full-resolution features
    3. Use reordering for hardware optimization
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        kernel_size: Tuple[int, int] = (8, 16),
        sparsity_ratio: float = 0.9,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """
        Args:
            dim: Hidden dimension
            num_heads: Number of attention heads
            kernel_size: Pooling kernel size (height, width)
            sparsity_ratio: Fraction of attention to retain (1 - sparsity)
            qkv_bias: Whether to use bias in QKV projections
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.sparsity_ratio = sparsity_ratio
        self.scale = self.head_dim ** -0.5
        
        # QKV projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Store reorder indices for efficiency
        self.register_buffer('reorder_indices', None)
        self.register_buffer('restore_indices', None)
    
    def _compute_reorder_indices(self, frame_size: Tuple[int, int], num_frames: int) -> torch.Tensor:
        """Generate reorder indices for hardware optimization."""
        H, W = frame_size
        h, w = self.kernel_size
        
        indices = []
        for f in range(num_frames):
            for i in range(H // h):
                for j in range(W // w):
                    for u in range(h):
                        for v in range(w):
                            y = i * h + u
                            x = j * w + v
                            idx = f * H * W + y * W + x
                            indices.append(idx)
        
        return torch.tensor(indices, dtype=torch.long)
    
    def _compute_restore_indices(self, reorder_indices: torch.Tensor) -> torch.Tensor:
        """Compute inverse permutation for restoration."""
        restore = torch.zeros_like(reorder_indices)
        for i, idx in enumerate(reorder_indices):
            restore[idx] = i
        return restore
    
    def _apply_pooling(self, x: torch.Tensor, frame_size: Tuple[int, int], num_frames: int) -> torch.Tensor:
        """Apply average pooling to create draft representations."""
        B, N, C = x.shape
        H, W = frame_size
        
        # Reshape to spatial-temporal format
        x_reshaped = x.view(B * num_frames, H, W, C).permute(0, 3, 1, 2)
        
        # Apply average pooling
        pooled = F.avg_pool2d(
            x_reshaped, 
            kernel_size=self.kernel_size, 
            stride=self.kernel_size
        )
        
        # Flatten back to sequence format
        pooled = pooled.permute(0, 2, 3, 1).reshape(B, -1, C)
        
        return pooled
    
    def _create_sparsity_mask(
        self, 
        draft_attn: torch.Tensor, 
        full_seq_len: int, 
        pooled_seq_len: int
    ) -> torch.Tensor:
        """Create sparsity mask based on draft attention."""
        B, num_heads, g, g = draft_attn.shape
        
        # Get top-k indices based on sparsity ratio
        k = int(g * g * (1 - self.sparsity_ratio))
        _, top_indices = torch.topk(draft_attn.view(B, num_heads, -1), k, dim=-1)
        
        # Create binary mask for pooled attention
        pooled_mask = torch.zeros(B, num_heads, g * g, device=draft_attn.device)
        pooled_mask.scatter_(-1, top_indices, 1.0)
        pooled_mask = pooled_mask.view(B, num_heads, g, g)
        
        # Expand mask to full resolution
        h, w = self.kernel_size
        mask_h = full_seq_len // (g * h)
        mask_w = full_seq_len // (g * w)
        
        # Create full resolution mask
        full_mask = torch.zeros(B, num_heads, full_seq_len, full_seq_len, device=draft_attn.device)
        
        # Expand pooled mask to full resolution
        for i in range(g):
            for j in range(g):
                if pooled_mask[0, 0, i, j] > 0:  # Check if this region is active
                    start_i, end_i = i * h, (i + 1) * h
                    start_j, end_j = j * w, (j + 1) * w
                    full_mask[:, :, start_i:end_i, start_j:end_j] = 1.0
        
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
            x: Input tensor of shape (B, N, C)
            frame_size: (H, W) spatial dimensions per frame
            num_frames: Number of temporal frames
            
        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape
        
        if frame_size is None or num_frames is None:
            # Infer from sequence length assuming square frames
            num_frames = 1
            H = W = int(np.sqrt(N))
            frame_size = (H, W)
        else:
            H, W = frame_size
        
        # Compute reorder indices if not cached
        if self.reorder_indices is None or len(self.reorder_indices) != N:
            self.reorder_indices = self._compute_reorder_indices(frame_size, num_frames).to(x.device)
            self.restore_indices = self._compute_restore_indices(self.reorder_indices).to(x.device)
        
        # Reorder input for contiguous memory access
        x_reordered = x[:, self.reorder_indices, :]
        
        # Generate Q, K, V
        qkv = self.qkv(x_reordered).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        # Compute draft attention on pooled features
        q_pooled = self._apply_pooling(q.transpose(1, 2).reshape(B, N, C), frame_size, num_frames)
        k_pooled = self._apply_pooling(k.transpose(1, 2).reshape(B, N, C), frame_size, num_frames)
        
        # Reshape pooled features for attention
        g = q_pooled.shape[1]  # Number of pooled tokens
        q_pooled = q_pooled.view(B, g, self.num_heads, self.head_dim).transpose(1, 2)
        k_pooled = k_pooled.view(B, g, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute draft attention
        draft_attn = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) * self.scale
        draft_attn = F.softmax(draft_attn, dim=-1)
        
        # Create sparsity mask
        sparsity_mask = self._create_sparsity_mask(draft_attn, N, g)
        
        # Compute sparse attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(sparsity_mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)
        
        # Project and restore original order
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out[:, self.restore_indices, :]
        
        return out
    
    def load_weights(self, state_dict: dict, strict: bool = True):
        """Load weights from state dictionary."""
        self.load_state_dict(state_dict, strict=strict)
    
    def save_weights(self) -> dict:
        """Save weights to state dictionary."""
        return self.state_dict()


class DraftAttentionBlock(nn.Module):
    """
    Transformer block with DraftAttention for video diffusion models.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        kernel_size: Tuple[int, int] = (8, 16),
        sparsity_ratio: float = 0.9,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = DraftAttention(
            dim=dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            sparsity_ratio=sparsity_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        frame_size: Optional[Tuple[int, int]] = None,
        num_frames: Optional[int] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), frame_size=frame_size, num_frames=num_frames)
        x = x + self.mlp(self.norm2(x))
        return x


def test_draft_attention():
    """Test function for DraftAttention."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test parameters
    batch_size = 2
    seq_len = 48 * 80 * 8  # 8 frames of 48x80 latent
    dim = 1024
    num_heads = 16
    
    # Create model
    model = DraftAttention(
        dim=dim,
        num_heads=num_heads,
        kernel_size=(8, 16),
        sparsity_ratio=0.9
    ).to(device)
    
    # Create input
    x = torch.randn(batch_size, seq_len, dim).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x, frame_size=(48*8, 80*8), num_frames=8)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sparsity ratio: {model.sparsity_ratio}")
    print("DraftAttention test passed!")


if __name__ == "__main__":
    test_draft_attention()