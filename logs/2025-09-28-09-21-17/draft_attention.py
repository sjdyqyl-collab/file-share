import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class DraftAttention(nn.Module):
    """
    DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance
    
    This implementation follows the paper's methodology:
    1. Create low-resolution draft attention via average pooling
    2. Generate sparsity pattern from draft attention
    3. Reorder tokens for hardware-friendly computation
    4. Apply sparse attention with restored order
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        sparsity_ratio: float = 0.9,
        pooling_kernel: Tuple[int, int] = (8, 16),
        block_size: int = 128,
        use_full_attention_steps: float = 0.25,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.sparsity_ratio = sparsity_ratio
        self.pooling_kernel = pooling_kernel
        self.block_size = block_size
        self.use_full_attention_steps = use_full_attention_steps
        
        # Validate dimensions
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Scale factor for attention
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def _compute_reorder_indices(
        self, 
        frame_size: Tuple[int, int], 
        num_frames: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate reorder and restore indices for patch-aligned processing."""
        H, W = frame_size
        h, w = self.pooling_kernel
        
        # Ensure dimensions are divisible
        assert H % h == 0 and W % w == 0, f"Frame size {frame_size} must be divisible by pooling kernel {self.pooling_kernel}"
        
        # Generate reorder indices
        reorder_indices = []
        for f in range(num_frames):
            for i in range(H // h):
                for j in range(W // w):
                    for u in range(h):
                        for v in range(w):
                            y = i * h + u
                            x = j * w + v
                            idx = f * H * W + y * W + x
                            reorder_indices.append(idx)
        
        reorder_indices = torch.tensor(reorder_indices, dtype=torch.long, device=device)
        
        # Generate restore indices (inverse permutation)
        restore_indices = torch.empty_like(reorder_indices)
        restore_indices[reorder_indices] = torch.arange(len(reorder_indices), device=device)
        
        return reorder_indices, restore_indices
    
    def _create_draft_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        frame_size: Tuple[int, int],
        num_frames: int
    ) -> torch.Tensor:
        """Create low-resolution draft attention via average pooling."""
        B, L, D = q.shape
        H, W = frame_size
        h, w = self.pooling_kernel
        
        # Reshape for pooling: (B, L, D) -> (B, num_frames, H, W, D)
        q_reshaped = q.view(B, num_frames, H, W, D)
        k_reshaped = k.view(B, num_frames, H, W, D)
        
        # Average pooling over spatial dimensions
        # Input: (B, num_frames, H, W, D)
        # Output: (B, num_frames, H//h, W//w, D)
        q_pooled = F.avg_pool2d(
            q_reshaped.permute(0, 1, 4, 2, 3).reshape(-1, D, H, W),
            kernel_size=self.pooling_kernel,
            stride=self.pooling_kernel
        ).view(B, num_frames, D, H//h, W//w).permute(0, 1, 3, 4, 2)
        
        k_pooled = F.avg_pool2d(
            k_reshaped.permute(0, 1, 4, 2, 3).reshape(-1, D, H, W),
            kernel_size=self.pooling_kernel,
            stride=self.pooling_kernel
        ).view(B, num_frames, D, H//h, W//w).permute(0, 1, 3, 4, 2)
        
        # Flatten pooled features: (B, num_frames * (H//h) * (W//w), D)
        g = num_frames * (H // h) * (W // w)
        q_draft = q_pooled.reshape(B, g, D)
        k_draft = k_pooled.reshape(B, g, D)
        
        # Compute draft attention: (B, g, g)
        draft_attention = torch.bmm(q_draft, k_draft.transpose(-2, -1)) * self.scale
        draft_attention = F.softmax(draft_attention, dim=-1)
        
        return draft_attention
    
    def _create_sparsity_mask(
        self,
        draft_attention: torch.Tensor,
        frame_size: Tuple[int, int],
        num_frames: int
    ) -> torch.Tensor:
        """Create token-level sparsity mask from draft attention."""
        B, g, _ = draft_attention.shape
        H, W = frame_size
        h, w = self.pooling_kernel
        
        # Calculate number of tokens
        n = num_frames * H * W
        
        # Sort draft attention scores and determine threshold
        flat_scores = draft_attention.view(B, -1)
        k = int(g * g * self.sparsity_ratio)
        
        # Get top-k indices
        _, top_indices = torch.topk(flat_scores, k=k, dim=-1)
        
        # Create region-level mask
        region_mask = torch.zeros_like(flat_scores)
        region_mask.scatter_(1, top_indices, 1.0)
        region_mask = region_mask.view(B, g, g)
        
        # Expand to token-level mask
        tokens_per_region = h * w
        token_mask = region_mask.repeat_interleave(tokens_per_region, dim=1)
        token_mask = token_mask.repeat_interleave(tokens_per_region, dim=2)
        
        # Ensure correct shape
        expected_shape = (B, n, n)
        if token_mask.shape != expected_shape:
            # Handle padding if needed
            pad_n = expected_shape[1] - token_mask.shape[1]
            if pad_n > 0:
                token_mask = F.pad(token_mask, (0, pad_n, 0, pad_n))
        
        return token_mask
    
    def forward(
        self,
        x: torch.Tensor,
        frame_size: Tuple[int, int],
        num_frames: int,
        timestep_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DraftAttention.
        
        Args:
            x: Input tensor of shape (B, L, D)
            frame_size: (H, W) spatial dimensions per frame
            num_frames: Number of frames
            timestep_ratio: Current timestep ratio (0.0 to 1.0) for progressive sparsity
        
        Returns:
            Output tensor of shape (B, L, D)
        """
        B, L, D = x.shape
        H, W = frame_size
        
        # Validate input dimensions
        expected_L = num_frames * H * W
        assert L == expected_L, f"Expected sequence length {expected_L}, got {L}"
        
        # Use full attention for initial steps
        if timestep_ratio is not None and timestep_ratio < self.use_full_attention_steps:
            return self._full_attention(x)
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (B, L, D)
        k = self.k_proj(x)  # (B, L, D)
        v = self.v_proj(x)  # (B, L, D)
        
        # Generate reorder indices
        reorder_indices, restore_indices = self._compute_reorder_indices(
            frame_size, num_frames, x.device
        )
        
        # Reorder tokens for patch-aligned processing
        q_reordered = q[:, reorder_indices, :]
        k_reordered = k[:, reorder_indices, :]
        v_reordered = v[:, reorder_indices, :]
        
        # Create draft attention and sparsity mask
        draft_attention = self._create_draft_attention(q_reordered, k_reordered, frame_size, num_frames)
        sparsity_mask = self._create_sparsity_mask(draft_attention, frame_size, num_frames)
        
        # Apply sparse attention
        output = self._sparse_attention(q_reordered, k_reordered, v_reordered, sparsity_mask)
        
        # Restore original order
        output_restored = torch.empty_like(output)
        output_restored[:, restore_indices, :] = output
        
        # Final projection
        return self.out_proj(output_restored)
    
    def _full_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Standard full attention computation."""
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        
        return self.out_proj(out)
    
    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply sparse attention with given mask."""
        B, L, D = q.shape
        
        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply sparsity mask
        mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        
        return out
    
    def load_weights(self, state_dict: dict):
        """Load pre-trained weights."""
        self.load_state_dict(state_dict)
    
    def get_sparsity_stats(self) -> dict:
        """Get sparsity statistics."""
        return {
            'sparsity_ratio': self.sparsity_ratio,
            'pooling_kernel': self.pooling_kernel,
            'block_size': self.block_size,
            'use_full_attention_steps': self.use_full_attention_steps
        }