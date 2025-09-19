"""
DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance
Implementation of the original paper method.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class DraftAttention(nn.Module):
    """
    Original DraftAttention implementation as described in the paper.
    
    This class implements the two-stage approach:
    1. Compute low-resolution draft attention map via down-sampling
    2. Use draft map to guide sparse attention computation at full resolution
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        sparsity_ratio: float = 0.9,
        pooling_kernel: Tuple[int, int] = (8, 16),
        use_full_attention_steps: int = 0,
        **kwargs
    ):
        """
        Initialize DraftAttention module.
        
        Args:
            dim: Hidden dimension size
            num_heads: Number of attention heads
            sparsity_ratio: Target sparsity ratio (fraction of connections to keep)
            pooling_kernel: (spatial, temporal) pooling kernel size
            use_full_attention_steps: Number of initial steps to use full attention
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sparsity_ratio = sparsity_ratio
        self.pooling_kernel = pooling_kernel
        self.use_full_attention_steps = use_full_attention_steps
        
        # Ensure dimension is divisible by num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Initialize weights
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
    
    def _compute_draft_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor,
        frame_size: Tuple[int, int],
        num_frames: int
    ) -> torch.Tensor:
        """
        Compute low-resolution draft attention map.
        
        Args:
            q: Query tensor [batch, seq_len, dim]
            k: Key tensor [batch, seq_len, dim]
            frame_size: (height, width) of each frame
            num_frames: Number of frames in the sequence
            
        Returns:
            Draft attention map [batch, g, g] where g is number of pooled regions
        """
        batch_size, seq_len, dim = q.shape
        height, width = frame_size
        
        # Calculate patch dimensions
        patch_h, patch_w = self.pooling_kernel
        num_patches_h = height // patch_h
        num_patches_w = width // patch_w
        
        # Total number of pooled regions
        g = num_frames * num_patches_h * num_patches_w
        
        # Reshape for pooling
        q_reshaped = q.view(batch_size, num_frames, height, width, dim)
        k_reshaped = k.view(batch_size, num_frames, height, width, dim)
        
        # Apply average pooling
        q_pooled = F.avg_pool3d(
            q_reshaped.permute(0, 4, 1, 2, 3),  # [B, dim, F, H, W]
            kernel_size=(1, patch_h, patch_w),
            stride=(1, patch_h, patch_w)
        ).permute(0, 2, 3, 4, 1).contiguous()  # [B, F', H', W', dim]
        
        k_pooled = F.avg_pool3d(
            k_reshaped.permute(0, 4, 1, 2, 3),
            kernel_size=(1, patch_h, patch_w),
            stride=(1, patch_h, patch_w)
        ).permute(0, 2, 3, 4, 1).contiguous()
        
        # Flatten spatial dimensions
        q_draft = q_pooled.view(batch_size, g, dim)
        k_draft = k_pooled.view(batch_size, g, dim)
        
        # Compute draft attention
        scale = 1.0 / np.sqrt(dim)
        attn_scores = torch.bmm(q_draft, k_draft.transpose(1, 2)) * scale
        attn_draft = F.softmax(attn_scores, dim=-1)
        
        return attn_draft
    
    def _generate_sparsity_mask(
        self,
        attn_draft: torch.Tensor,
        seq_len: int,
        frame_size: Tuple[int, int],
        num_frames: int
    ) -> torch.Tensor:
        """
        Generate sparsity mask based on draft attention.
        
        Args:
            attn_draft: Draft attention map [batch, g, g]
            seq_len: Original sequence length
            frame_size: (height, width) of each frame
            num_frames: Number of frames
            
        Returns:
            Binary sparsity mask [batch, seq_len, seq_len]
        """
        batch_size, g, _ = attn_draft.shape
        height, width = frame_size
        
        # Calculate number of connections to keep
        num_keep = int(g * g * (1 - self.sparsity_ratio))
        
        # Get top-k indices for each batch
        mask_draft = torch.zeros_like(attn_draft)
        
        for b in range(batch_size):
            # Flatten attention scores
            flat_attn = attn_draft[b].flatten()
            
            # Get top-k values and indices
            _, top_indices = torch.topk(flat_attn, num_keep)
            
            # Convert flat indices back to 2D
            row_indices = top_indices // g
            col_indices = top_indices % g
            
            # Set mask values
            mask_draft[b, row_indices, col_indices] = 1.0
        
        # Calculate patch dimensions
        patch_h, patch_w = self.pooling_kernel
        num_patches_h = height // patch_h
        num_patches_w = width // patch_w
        
        # Expand mask to full resolution
        mask_full = torch.zeros(batch_size, seq_len, seq_len, device=attn_draft.device)
        
        for b in range(batch_size):
            for i in range(g):
                for j in range(g):
                    if mask_draft[b, i, j] > 0:
                        # Calculate corresponding token ranges
                        frame_i = i // (num_patches_h * num_patches_w)
                        patch_i = i % (num_patches_h * num_patches_w)
                        patch_row_i = patch_i // num_patches_w
                        patch_col_i = patch_i % num_patches_w
                        
                        frame_j = j // (num_patches_h * num_patches_w)
                        patch_j = j % (num_patches_h * num_patches_w)
                        patch_row_j = patch_j // num_patches_w
                        patch_col_j = patch_j % num_patches_w
                        
                        # Map to token indices
                        start_i = frame_i * height * width + patch_row_i * patch_h * width + patch_col_i * patch_w
                        end_i = start_i + patch_h * patch_w
                        
                        start_j = frame_j * height * width + patch_row_j * patch_h * width + patch_col_j * patch_w
                        end_j = start_j + patch_h * patch_w
                        
                        mask_full[b, start_i:end_i, start_j:end_j] = 1.0
        
        return mask_full
    
    def _reorder_tokens(self, x: torch.Tensor, frame_size: Tuple[int, int], num_frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reorder tokens for efficient memory access.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            frame_size: (height, width) of each frame
            num_frames: Number of frames
            
        Returns:
            reordered_x: Reordered tensor
            restore_indices: Indices to restore original order
        """
        batch_size, seq_len, dim = x.shape
        height, width = frame_size
        
        # Create reordering indices based on spatial locality
        patch_h, patch_w = self.pooling_kernel
        
        # Generate indices for spatial grouping
        indices = []
        for f in range(num_frames):
            for ph in range(0, height, patch_h):
                for pw in range(0, width, patch_w):
                    # Add all tokens in this patch
                    for h in range(ph, min(ph + patch_h, height)):
                        for w in range(pw, min(pw + patch_w, width)):
                            idx = f * height * width + h * width + w
                            indices.append(idx)
        
        indices = torch.tensor(indices, device=x.device, dtype=torch.long)
        
        # Reorder tokens
        reordered_x = x[:, indices, :]
        
        # Create restore indices (inverse permutation)
        restore_indices = torch.empty_like(indices)
        restore_indices[indices] = torch.arange(seq_len, device=x.device)
        
        return reordered_x, restore_indices
    
    def forward(
        self, 
        x: torch.Tensor,
        frame_size: Tuple[int, int],
        num_frames: int,
        step_idx: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of DraftAttention.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            frame_size: (height, width) of each frame
            num_frames: Number of frames
            step_idx: Current denoising step (for scheduling)
            
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        batch_size, seq_len, dim = x.shape
        
        # Use full attention for initial steps
        if step_idx < self.use_full_attention_steps:
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Standard attention
            scale = 1.0 / np.sqrt(self.head_dim)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn_weights, v)
            
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
            return self.out_proj(out)
        
        # Reorder tokens for efficient memory access
        x_reordered, restore_indices = self._reorder_tokens(x, frame_size, num_frames)
        
        # Linear projections
        q = self.q_proj(x_reordered)
        k = self.k_proj(x_reordered)
        v = self.v_proj(x_reordered)
        
        # Compute draft attention
        attn_draft = self._compute_draft_attention(q, k, frame_size, num_frames)
        
        # Generate sparsity mask
        sparsity_mask = self._generate_sparsity_mask(attn_draft, seq_len, frame_size, num_frames)
        
        # Multi-head attention with sparsity
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention with sparsity mask
        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply sparsity mask
        sparsity_mask = sparsity_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn_scores = attn_scores.masked_fill(sparsity_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        out = self.out_proj(out)
        
        # Restore original token order
        out_restored = out[:, restore_indices, :]
        
        return out_restored
    
    def load_weights(self, checkpoint_path: str):
        """Load pretrained weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint)
        print(f"Loaded weights from {checkpoint_path}")
    
    def save_weights(self, checkpoint_path: str):
        """Save current weights to checkpoint."""
        torch.save(self.state_dict(), checkpoint_path)
        print(f"Saved weights to {checkpoint_path}")


# Utility functions for testing
def test_draft_attention():
    """Test function for DraftAttention."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test configuration
    batch_size = 2
    seq_len = 128 * 64  # 128 frames, 64x64 patches
    dim = 512
    num_heads = 8
    frame_size = (64, 64)
    num_frames = 128
    
    # Create model
    model = DraftAttention(
        dim=dim,
        num_heads=num_heads,
        sparsity_ratio=0.9,
        pooling_kernel=(8, 16)
    ).to(device)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, dim).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x, frame_size=frame_size, num_frames=num_frames, step_idx=50)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model, output


if __name__ == "__main__":
    test_draft_attention()