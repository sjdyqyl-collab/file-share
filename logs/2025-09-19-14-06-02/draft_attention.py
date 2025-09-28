"""
DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance
Implementation of the original paper method.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class DraftAttention(nn.Module):
    """
    Original DraftAttention implementation as described in the paper.
    
    This module implements the two-stage approach:
    1. Draft stage: Compute low-resolution attention on downsampled representations
    2. Sparse stage: Apply guided sparsity on full resolution
    
    Args:
        sparsity_ratio: Target sparsity ratio (e.g., 0.9 for 90% sparsity)
        kernel_size: Pooling kernel size (default 8x16 as in paper)
        stride: Pooling stride (default same as kernel_size)
        full_attention_steps: Number of initial steps to use full attention
    """
    
    def __init__(
        self,
        sparsity_ratio: float = 0.9,
        kernel_size: Tuple[int, int] = (8, 16),
        stride: Optional[Tuple[int, int]] = None,
        full_attention_steps: int = 0,
    ):
        super().__init__()
        self.sparsity_ratio = sparsity_ratio
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.full_attention_steps = full_attention_steps
        self.current_step = 0
        
    def reset_step(self):
        """Reset the current step counter."""
        self.current_step = 0
        
    def _compute_reorder_indices(
        self, 
        frame_size: Tuple[int, int], 
        patch_size: Tuple[int, int], 
        num_frames: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate reorder and restore indices for hardware optimization.
        
        Args:
            frame_size: (H, W) spatial dimensions
            patch_size: (h, w) patch dimensions for pooling
            num_frames: Number of temporal frames
            
        Returns:
            reorder_indices: Tensor of shape [n] for reordering
            restore_indices: Tensor of shape [n] for restoring original order
        """
        H, W = frame_size
        h, w = patch_size
        n = num_frames * H * W
        
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
        
        reorder_indices = torch.tensor(reorder_indices, dtype=torch.long)
        
        # Generate restore indices (inverse permutation)
        restore_indices = torch.empty_like(reorder_indices)
        restore_indices[reorder_indices] = torch.arange(n, dtype=torch.long)
        
        return reorder_indices, restore_indices
    
    def _downsample_with_pooling(
        self, 
        x: torch.Tensor, 
        frame_size: Tuple[int, int], 
        num_frames: int
    ) -> torch.Tensor:
        """
        Downsample tokens using average pooling.
        
        Args:
            x: Input tensor of shape [B, n, d] where n = F*H*W
            frame_size: (H, W) spatial dimensions
            num_frames: Number of temporal frames
            
        Returns:
            Downsampled tensor of shape [B, g, d] where g = F*(H//h)*(W//w)
        """
        B, n, d = x.shape
        H, W = frame_size
        
        # Reshape to spatial-temporal format
        x = x.view(B, num_frames, H, W, d)
        x = x.permute(0, 1, 4, 2, 3)  # [B, F, d, H, W]
        
        # Apply average pooling
        pooled = F.avg_pool2d(
            x.view(B * num_frames, d, H, W),
            kernel_size=self.kernel_size,
            stride=self.stride
        )
        
        # Reshape back to sequence format
        pooled = pooled.view(B, num_frames, d, -1)
        pooled = pooled.permute(0, 1, 3, 2)  # [B, F, g_per_frame, d]
        
        # Flatten spatial dimensions
        g = pooled.shape[2]
        pooled = pooled.view(B, -1, d)  # [B, F*g_per_frame, d]
        
        return pooled
    
    def _compute_draft_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute draft attention on downsampled representations.
        
        Args:
            q, k, v: Query, key, value tensors [B, g, d]
            
        Returns:
            Draft attention map [B, g, g]
        """
        d = q.shape[-1]
        scores = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(d)
        attention = F.softmax(scores, dim=-1)
        return attention
    
    def _generate_sparsity_mask(
        self, 
        draft_attention: torch.Tensor, 
        sparsity_ratio: float
    ) -> torch.Tensor:
        """
        Generate sparsity mask from draft attention.
        
        Args:
            draft_attention: [B, g, g] draft attention map
            sparsity_ratio: Target sparsity ratio
            
        Returns:
            Binary mask [B, g, g] indicating allowed attention patterns
        """
        B, g, _ = draft_attention.shape
        
        # Flatten and find threshold for top-r values
        flat_attention = draft_attention.view(B, -1)
        k = int(sparsity_ratio * g * g)
        
        # Get top-k values and their indices
        _, top_indices = torch.topk(flat_attention, k, dim=-1)
        
        # Create binary mask
        mask = torch.zeros_like(flat_attention)
        mask.scatter_(1, top_indices, 1.0)
        mask = mask.view(B, g, g)
        
        return mask
    
    def _extend_mask_to_full_resolution(
        self, 
        mask: torch.Tensor, 
        original_size: int,
        frame_size: Tuple[int, int],
        num_frames: int
    ) -> torch.Tensor:
        """
        Extend region-level mask to full token-level mask.
        
        Args:
            mask: [B, g, g] region-level mask
            original_size: Original sequence length n
            frame_size: (H, W) spatial dimensions
            num_frames: Number of temporal frames
            
        Returns:
            Token-level mask [B, n, n]
        """
        B, g, _ = mask.shape
        H, W = frame_size
        h, w = self.kernel_size
        
        # Calculate tokens per region
        tokens_per_region = h * w
        tokens_per_frame = H * W
        regions_per_frame = (H // h) * (W // w)
        
        # Create token-level mask
        full_mask = torch.zeros(B, original_size, original_size, 
                              device=mask.device, dtype=mask.dtype)
        
        for b in range(B):
            for f1 in range(num_frames):
                for f2 in range(num_frames):
                    for i in range(regions_per_frame):
                        for j in range(regions_per_frame):
                            if mask[b, f1 * regions_per_frame + i, 
                                   f2 * regions_per_frame + j] > 0:
                                
                                # Map region indices to token indices
                                region_i_start = f1 * tokens_per_frame + (i // (W // w)) * h * W + (i % (W // w)) * w
                                region_j_start = f2 * tokens_per_frame + (j // (W // w)) * h * W + (j % (W // w)) * w
                                
                                for u in range(h):
                                    for v in range(w):
                                        token_i = region_i_start + u * W + v
                                        for x in range(h):
                                            for y in range(w):
                                                token_j = region_j_start + x * W + y
                                                if token_i < original_size and token_j < original_size:
                                                    full_mask[b, token_i, token_j] = 1.0
        
        return full_mask
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        frame_size: Tuple[int, int],
        num_frames: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of DraftAttention.
        
        Args:
            q, k, v: Query, key, value tensors [B, n, d]
            frame_size: (H, W) spatial dimensions
            num_frames: Number of temporal frames
            attention_mask: Optional attention mask [B, n, n]
            
        Returns:
            Output tensor [B, n, d]
        """
        B, n, d = q.shape
        
        # Use full attention for initial steps if specified
        if self.current_step < self.full_attention_steps:
            self.current_step += 1
            scores = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(d)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float('-inf'))
            attention = F.softmax(scores, dim=-1)
            return torch.bmm(attention, v)
        
        self.current_step += 1
        
        # Stage 1: Draft attention computation
        q_draft = self._downsample_with_pooling(q, frame_size, num_frames)
        k_draft = self._downsample_with_pooling(k, frame_size, num_frames)
        
        draft_attention = self._compute_draft_attention(q_draft, k_draft, k_draft)
        
        # Stage 2: Generate sparsity mask
        sparsity_mask = self._generate_sparsity_mask(
            draft_attention, 
            self.sparsity_ratio
        )
        
        # Extend to full resolution
        full_sparsity_mask = self._extend_mask_to_full_resolution(
            sparsity_mask,
            n,
            frame_size,
            num_frames
        )
        
        # Apply sparsity mask
        scores = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(d)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply sparsity mask
        scores = scores.masked_fill(full_sparsity_mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        
        # Ensure sparsity is maintained
        attention = attention * full_sparsity_mask
        
        return torch.bmm(attention, v)
    
    def load_weights(self, state_dict: dict):
        """Load weights if available (no trainable parameters in base version)."""
        pass
    
    def save_weights(self) -> dict:
        """Save weights (empty dict for base version)."""
        return {}


class DraftAttentionBlock(nn.Module):
    """
    Multi-head attention block using DraftAttention.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        sparsity_ratio: float = 0.9,
        kernel_size: Tuple[int, int] = (8, 16),
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.draft_attention = DraftAttention(
            sparsity_ratio=sparsity_ratio,
            kernel_size=kernel_size
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        frame_size: Tuple[int, int],
        num_frames: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, n, d_model]
            frame_size: (H, W) spatial dimensions
            num_frames: Number of temporal frames
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [B, n, d_model]
        """
        B, n, d = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply draft attention per head
        out = []
        for head_idx in range(self.n_heads):
            head_q = q[:, head_idx]  # [B, n, d_head]
            head_k = k[:, head_idx]  # [B, n, d_head]
            head_v = v[:, head_idx]  # [B, n, d_head]
            
            head_out = self.draft_attention(
                head_q, head_k, head_v,
                frame_size, num_frames, attention_mask
            )
            out.append(head_out)
        
        # Concatenate heads
        out = torch.stack(out, dim=1)  # [B, n_heads, n, d_head]
        out = out.transpose(1, 2).contiguous().view(B, n, d)
        
        # Final projection
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out