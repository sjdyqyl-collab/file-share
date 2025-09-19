"""
DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance
Implementation of the core DraftAttention method as described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np


class DraftAttention(nn.Module):
    """
    Training-free sparse attention mechanism that uses low-resolution draft attention
    maps to guide sparse computation in full resolution.
    
    Args:
        sparsity_ratio: Fraction of tokens to keep in sparse attention (0.5-0.95)
        pooling_kernel: Tuple of (temporal, spatial) pooling kernel sizes
        fallback_steps: Number of initial denoising steps to use full attention
        device: Target device for computation
    """
    
    def __init__(
        self,
        sparsity_ratio: float = 0.75,
        pooling_kernel: Tuple[int, int] = (8, 16),
        fallback_steps: int = 5,
        device: str = "cuda"
    ):
        super().__init__()
        self.sparsity_ratio = sparsity_ratio
        self.pooling_kernel = pooling_kernel
        self.fallback_steps = fallback_steps
        self.device = device
        
        # Calculate token reduction factor
        self.temporal_reduction = pooling_kernel[0]
        self.spatial_reduction = pooling_kernel[1]
        self.total_reduction = self.temporal_reduction * self.spatial_reduction
        
        # Initialize reordering indices cache
        self._reordering_cache = {}
        
    def _compute_draft_attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute low-resolution draft attention map.
        
        Args:
            query: [B, n_heads, T*H*W, d_head]
            key: [B, n_heads, T*H*W, d_head]
            value: [B, n_heads, T*H*W, d_head]
            
        Returns:
            draft_attention: [B, n_heads, g, g] where g = T*H*W / total_reduction
            draft_indices: Indices for reordering tokens
        """
        B, n_heads, seq_len, d_head = query.shape
        
        # Reshape for pooling: [B, n_heads, T, H, W, d_head]
        T = seq_len // (self.spatial_reduction * self.spatial_reduction)
        H = W = int(np.sqrt(seq_len // T))
        
        # Ensure dimensions are divisible by pooling kernels
        T_pooled = T // self.temporal_reduction
        H_pooled = H // self.spatial_reduction
        W_pooled = W // self.spatial_reduction
        
        # Reshape tensors for pooling
        query_reshaped = query.view(B, n_heads, T, H, W, d_head)
        key_reshaped = key.view(B, n_heads, T, H, W, d_head)
        
        # Apply average pooling to create draft queries and keys
        draft_query = F.avg_pool3d(
            query_reshaped.permute(0, 1, 5, 2, 3, 4).reshape(-1, d_head, T, H, W),
            kernel_size=(self.temporal_reduction, self.spatial_reduction, self.spatial_reduction),
            stride=(self.temporal_reduction, self.spatial_reduction, self.spatial_reduction)
        ).view(B, n_heads, d_head, T_pooled, H_pooled, W_pooled).permute(0, 1, 3, 4, 5, 2)
        
        draft_key = F.avg_pool3d(
            key_reshaped.permute(0, 1, 5, 2, 3, 4).reshape(-1, d_head, T, H, W),
            kernel_size=(self.temporal_reduction, self.spatial_reduction, self.spatial_reduction),
            stride=(self.temporal_reduction, self.spatial_reduction, self.spatial_reduction)
        ).view(B, n_heads, d_head, T_pooled, H_pooled, W_pooled).permute(0, 1, 3, 4, 5, 2)
        
        # Compute draft attention
        draft_query = draft_query.reshape(B, n_heads, -1, d_head)
        draft_key = draft_key.reshape(B, n_heads, -1, d_head)
        
        # Scale for numerical stability
        scale = 1.0 / np.sqrt(d_head)
        draft_attention = torch.matmul(draft_query, draft_key.transpose(-2, -1)) * scale
        
        return draft_attention, (T_pooled, H_pooled, W_pooled)
    
    def _create_sparsity_mask(
        self, 
        draft_attention: torch.Tensor,
        sparsity_ratio: Optional[float] = None
    ) -> torch.Tensor:
        """
        Create binary mask from draft attention map.
        
        Args:
            draft_attention: [B, n_heads, g, g] draft attention scores
            sparsity_ratio: Override default sparsity ratio
            
        Returns:
            mask: [B, n_heads, seq_len] binary mask for sparse attention
        """
        if sparsity_ratio is None:
            sparsity_ratio = self.sparsity_ratio
            
        B, n_heads, g, _ = draft_attention.shape
        
        # Upsample draft attention to full resolution
        # First, compute attention scores per draft block
        draft_scores = draft_attention.mean(dim=-1)  # [B, n_heads, g]
        
        # Upsample to full resolution using nearest neighbor
        full_scores = F.interpolate(
            draft_scores.unsqueeze(-1).view(B * n_heads, g, 1, 1),
            size=(self.total_reduction, 1),
            mode='nearest'
        ).view(B, n_heads, -1)
        
        # Create mask by selecting top-k tokens
        k = int(full_scores.shape[-1] * sparsity_ratio)
        _, top_indices = torch.topk(full_scores, k=k, dim=-1, sorted=False)
        
        mask = torch.zeros_like(full_scores, dtype=torch.bool)
        mask.scatter_(-1, top_indices, True)
        
        return mask
    
    def _reorder_tokens(
        self, 
        tensor: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reorder tokens based on sparsity mask for efficient computation.
        
        Args:
            tensor: [B, n_heads, seq_len, d_head]
            mask: [B, n_heads, seq_len] binary mask
            
        Returns:
            reordered_tensor: [B, n_heads, k, d_head] where k = seq_len * sparsity_ratio
            restore_indices: Indices to restore original order
        """
        B, n_heads, seq_len, d_head = tensor.shape
        
        # Get indices for selected tokens
        selected_indices = mask.nonzero(as_tuple=True)
        
        # Create reordering indices
        batch_indices = selected_indices[0]
        head_indices = selected_indices[1]
        token_indices = selected_indices[2]
        
        # Reorder tensor
        reordered = tensor[batch_indices, head_indices, token_indices, :]
        
        # Reshape to [B, n_heads, k, d_head]
        k = int(seq_len * self.sparsity_ratio)
        reordered = reordered.view(B, n_heads, k, d_head)
        
        # Store restore indices
        restore_indices = (batch_indices, head_indices, token_indices)
        
        return reordered, restore_indices
    
    def _restore_order(
        self, 
        sparse_output: torch.Tensor, 
        restore_indices: Tuple[torch.Tensor, ...],
        original_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Restore original token order after sparse attention computation.
        
        Args:
            sparse_output: [B, n_heads, k, d_head] sparse attention output
            restore_indices: Indices from _reorder_tokens
            original_shape: Original tensor shape
            
        Returns:
            restored: [B, n_heads, seq_len, d_head] restored output
        """
        B, n_heads, seq_len, d_head = original_shape
        
        # Initialize output tensor
        restored = torch.zeros(original_shape, dtype=sparse_output.dtype, device=sparse_output.device)
        
        # Scatter sparse output back to original positions
        batch_indices, head_indices, token_indices = restore_indices
        restored[batch_indices, head_indices, token_indices, :] = sparse_output.view(-1, d_head)
        
        return restored
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        step: int = 0,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of DraftAttention.
        
        Args:
            query: [B, n_heads, seq_len, d_head]
            key: [B, n_heads, seq_len, d_head]
            value: [B, n_heads, seq_len, d_head]
            step: Current denoising step (for fallback)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing output and optionally attention weights
        """
        B, n_heads, seq_len, d_head = query.shape
        
        # Use dense attention for initial fallback steps
        if step < self.fallback_steps:
            # Standard scaled dot-product attention
            scale = 1.0 / np.sqrt(d_head)
            attention = torch.matmul(query, key.transpose(-2, -1)) * scale
            attention = F.softmax(attention, dim=-1)
            output = torch.matmul(attention, value)
            
            return {"output": output, "attention": attention if return_attention else None}
        
        # Compute draft attention
        draft_attention, _ = self._compute_draft_attention(query, key, value)
        
        # Create sparsity mask
        mask = self._create_sparsity_mask(draft_attention)
        
        # Reorder tokens for efficient computation
        reordered_query, query_indices = self._reorder_tokens(query, mask)
        reordered_key, key_indices = self._reorder_tokens(key, mask)
        reordered_value, value_indices = self._reorder_tokens(value, mask)
        
        # Compute sparse attention
        scale = 1.0 / np.sqrt(d_head)
        sparse_attention = torch.matmul(reordered_query, reordered_key.transpose(-2, -1)) * scale
        sparse_attention = F.softmax(sparse_attention, dim=-1)
        sparse_output = torch.matmul(sparse_attention, reordered_value)
        
        # Restore original order
        output = self._restore_order(sparse_output, value_indices, (B, n_heads, seq_len, d_head))
        
        result = {"output": output}
        
        if return_attention:
            # Create full attention matrix from sparse attention
            full_attention = torch.zeros(B, n_heads, seq_len, seq_len, 
                                       dtype=sparse_attention.dtype, device=sparse_attention.device)
            
            # This is a simplified reconstruction - in practice, you'd need more sophisticated mapping
            result["attention"] = full_attention
        
        return result
    
    def load_weights(self, checkpoint_path: str) -> None:
        """
        Load pre-trained weights if available.
        Note: DraftAttention is training-free, so this is mainly for compatibility.
        """
        # No weights to load for training-free method
        pass
    
    def save_weights(self, checkpoint_path: str) -> None:
        """
        Save current state (mainly for configuration).
        """
        torch.save({
            'sparsity_ratio': self.sparsity_ratio,
            'pooling_kernel': self.pooling_kernel,
            'fallback_steps': self.fallback_steps,
            'config': self.__dict__
        }, checkpoint_path)


class DraftAttentionConfig:
    """Configuration class for DraftAttention hyperparameters."""
    
    def __init__(
        self,
        sparsity_ratio: float = 0.75,
        pooling_kernel: Tuple[int, int] = (8, 16),
        fallback_steps: int = 5,
        device: str = "cuda"
    ):
        self.sparsity_ratio = sparsity_ratio
        self.pooling_kernel = pooling_kernel
        self.fallback_steps = fallback_steps
        self.device = device
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sparsity_ratio': self.sparsity_ratio,
            'pooling_kernel': self.pooling_kernel,
            'fallback_steps': self.fallback_steps,
            'device': self.device
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DraftAttentionConfig':
        return cls(**config_dict)