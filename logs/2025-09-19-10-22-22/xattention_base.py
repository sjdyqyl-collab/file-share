"""
XAttention: Sparse Attention with Antidiagonal Pattern Selection
Baseline Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class XAttentionBase(nn.Module):
    """
    Baseline XAttention implementation with antidiagonal pattern selection.
    
    This implements the core XAttention method from the paper:
    - Strided antidiagonal scoring for block importance
    - Threshold-based block selection
    - Sparse attention computation
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int = 8,
        stride: int = 8,
        threshold: float = 0.9,
        dropout: float = 0.0,
        use_dynamic_threshold: bool = False,
        max_threshold_adjustments: int = 1000,
    ):
        """
        Initialize XAttention.
        
        Args:
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
            block_size: Size of attention blocks (B×B)
            stride: Stride for antidiagonal sampling (S)
            threshold: Selection threshold (τ)
            dropout: Dropout probability
            use_dynamic_threshold: Whether to use dynamic threshold optimization
            max_threshold_adjustments: Maximum threshold adjustments for DP
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size
        self.stride = stride
        self.threshold = threshold
        self.use_dynamic_threshold = use_dynamic_threshold
        self.max_threshold_adjustments = max_threshold_adjustments
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Dynamic threshold optimization state
        if use_dynamic_threshold:
            self.register_buffer('threshold_per_head', torch.ones(num_heads) * threshold)
            self.register_buffer('threshold_history', torch.zeros(num_heads, max_threshold_adjustments))
            self.register_buffer('performance_history', torch.zeros(num_heads, max_threshold_adjustments))
            self.register_buffer('adjustment_count', torch.zeros(num_heads, dtype=torch.long))
    
    def reshape_for_antidiagonal(self, x: torch.Tensor, stride: int) -> torch.Tensor:
        """
        Reshape tensor for antidiagonal pattern processing.
        
        Args:
            x: Input tensor of shape [batch, seq_len, head_dim]
            stride: Stride for antidiagonal sampling
            
        Returns:
            Reshaped tensor for antidiagonal processing
        """
        batch_size, seq_len, head_dim = x.shape
        
        # Create antidiagonal indices
        indices = []
        for diag_offset in range(-seq_len + 1, seq_len):
            if (diag_offset % stride) == 0:
                row_indices = []
                col_indices = []
                for i in range(max(0, -diag_offset), min(seq_len, seq_len - diag_offset)):
                    j = i + diag_offset
                    if 0 <= j < seq_len:
                        row_indices.append(i)
                        col_indices.append(j)
                if row_indices:
                    indices.append((row_indices, col_indices))
        
        # Extract antidiagonal elements
        antidiag_features = []
        for row_idx, col_idx in indices:
            antidiag_features.append(x[:, row_idx, :])
        
        if antidiag_features:
            return torch.cat(antidiag_features, dim=1)
        else:
            return x
    
    def compute_block_importance(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        block_start: int, 
        block_end: int
    ) -> torch.Tensor:
        """
        Compute importance score for a block using antidiagonal patterns.
        
        Args:
            q: Query tensor [batch, seq_len, head_dim]
            k: Key tensor [batch, seq_len, head_dim]
            block_start: Start index of block
            block_end: End index of block
            
        Returns:
            Block importance score
        """
        # Extract block
        q_block = q[:, block_start:block_end, :]
        k_block = k[:, block_start:block_end, :]
        
        # Reshape for antidiagonal processing
        q_reshaped = self.reshape_for_antidiagonal(q_block, self.stride)
        k_reshaped = self.reshape_for_antidiagonal(k_block, self.stride)
        
        # Compute approximate attention
        if q_reshaped.shape[1] > 0 and k_reshaped.shape[1] > 0:
            # Scale factor for antidiagonal approximation
            scale = 1.0 / np.sqrt(self.head_dim * self.stride)
            
            # Compute attention scores on antidiagonal elements
            attn_scores = torch.bmm(q_reshaped, k_reshaped.transpose(1, 2)) * scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # Sum all attention weights as block importance
            importance = attn_weights.sum(dim=(1, 2))
        else:
            # Fallback: use L2 norm as importance
            importance = torch.norm(q_block, dim=(1, 2)) + torch.norm(k_block, dim=(1, 2))
        
        return importance
    
    def select_important_blocks(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        seq_len: int,
        head_idx: int
    ) -> torch.Tensor:
        """
        Select important blocks based on threshold.
        
        Args:
            q: Query tensor
            k: Key tensor
            seq_len: Sequence length
            head_idx: Head index for dynamic threshold
            
        Returns:
            Binary mask indicating selected blocks
        """
        batch_size = q.shape[0]
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        # Compute importance for each block
        block_importance = []
        for b in range(num_blocks):
            block_start = b * self.block_size
            block_end = min((b + 1) * self.block_size, seq_len)
            
            importance = self.compute_block_importance(q, k, block_start, block_end)
            block_importance.append(importance)
        
        # Stack importance scores
        importance_scores = torch.stack(block_importance, dim=1)  # [batch, num_blocks]
        
        # Apply softmax normalization
        importance_probs = F.softmax(importance_scores, dim=1)
        
        # Get current threshold
        if self.use_dynamic_threshold and hasattr(self, 'threshold_per_head'):
            current_threshold = self.threshold_per_head[head_idx].item()
        else:
            current_threshold = self.threshold
        
        # Select blocks greedily until threshold is met
        selected_blocks = torch.zeros_like(importance_probs)
        
        for batch_idx in range(batch_size):
            cumulative_importance = 0.0
            # Sort blocks by importance
            sorted_indices = torch.argsort(importance_probs[batch_idx], descending=True)
            
            for block_idx in sorted_indices:
                if cumulative_importance >= current_threshold:
                    break
                selected_blocks[batch_idx, block_idx] = 1.0
                cumulative_importance += importance_probs[batch_idx, block_idx]
        
        return selected_blocks
    
    def create_sparse_mask(
        self, 
        selected_blocks: torch.Tensor, 
        seq_len: int
    ) -> torch.Tensor:
        """
        Create sparse attention mask from selected blocks.
        
        Args:
            selected_blocks: Binary block selection mask
            seq_len: Sequence length
            
        Returns:
            Sparse attention mask
        """
        batch_size = selected_blocks.shape[0]
        mask = torch.zeros(batch_size, seq_len, seq_len, device=selected_blocks.device)
        
        num_blocks = selected_blocks.shape[1]
        
        for b in range(num_blocks):
            block_start = b * self.block_size
            block_end = min((b + 1) * self.block_size, seq_len)
            
            # Mark selected blocks in mask
            for batch_idx in range(batch_size):
                if selected_blocks[batch_idx, b] > 0.5:
                    mask[batch_idx, block_start:block_end, block_start:block_end] = 1.0
        
        return mask
    
    def optimize_threshold(
        self, 
        head_idx: int, 
        current_performance: float
    ):
        """
        Optimize threshold using dynamic programming approach.
        
        Args:
            head_idx: Head index
            current_performance: Current performance metric
        """
        if not self.use_dynamic_threshold:
            return
        
        adjustment_count = self.adjustment_count[head_idx].item()
        
        if adjustment_count < self.max_threshold_adjustments:
            # Store history
            self.threshold_history[head_idx, adjustment_count] = self.threshold_per_head[head_idx]
            self.performance_history[head_idx, adjustment_count] = current_performance
            
            # Simple threshold update (reduce by 0.9 factor if performance drops)
            if adjustment_count > 0:
                prev_performance = self.performance_history[head_idx, adjustment_count - 1]
                if current_performance < prev_performance * 0.95:
                    self.threshold_per_head[head_idx] *= 0.9
                    self.threshold_per_head[head_idx] = max(0.1, self.threshold_per_head[head_idx])
            
            self.adjustment_count[head_idx] += 1
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of XAttention.
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            head_mask: Optional head mask
            output_attentions: Whether to output attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Process each head separately
        outputs = []
        attention_weights_list = []
        
        for head_idx in range(self.num_heads):
            q_head = q[:, head_idx, :, :]  # [batch, seq_len, head_dim]
            k_head = k[:, head_idx, :, :]
            v_head = v[:, head_idx, :, :]
            
            # Select important blocks
            selected_blocks = self.select_important_blocks(q_head, k_head, seq_len, head_idx)
            
            # Create sparse mask
            sparse_mask = self.create_sparse_mask(selected_blocks, seq_len)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                sparse_mask = sparse_mask * attention_mask
            
            # Compute attention scores
            scale = 1.0 / np.sqrt(self.head_dim)
            attn_scores = torch.bmm(q_head, k_head.transpose(1, 2)) * scale
            
            # Apply sparse mask
            attn_scores = attn_scores.masked_fill(sparse_mask == 0, float('-inf'))
            
            # Compute attention weights
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            attn_output = torch.bmm(attn_weights, v_head)
            
            outputs.append(attn_output)
            if output_attentions:
                attention_weights_list.append(attn_weights)
        
        # Concatenate heads
        attn_output = torch.stack(outputs, dim=1)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Final projection
        attn_output = self.out_proj(attn_output)
        
        # Prepare attention weights output
        if output_attentions:
            attention_weights = torch.stack(attention_weights_list, dim=1)
        else:
            attention_weights = None
        
        return attn_output, attention_weights
    
    def get_sparsity_stats(self) -> dict:
        """Get sparsity statistics."""
        stats = {
            'block_size': self.block_size,
            'stride': self.stride,
            'threshold': self.threshold,
            'use_dynamic_threshold': self.use_dynamic_threshold,
        }
        
        if self.use_dynamic_threshold and hasattr(self, 'threshold_per_head'):
            stats['threshold_per_head'] = self.threshold_per_head.cpu().numpy().tolist()
            stats['adjustment_count'] = self.adjustment_count.cpu().numpy().tolist()
        
        return stats


class XAttentionConfig:
    """Configuration class for XAttention."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        block_size: int = 8,
        stride: int = 8,
        threshold: float = 0.9,
        dropout: float = 0.0,
        use_dynamic_threshold: bool = False,
        max_threshold_adjustments: int = 1000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.block_size = block_size
        self.stride = stride
        self.threshold = threshold
        self.dropout = dropout
        self.use_dynamic_threshold = use_dynamic_threshold
        self.max_threshold_adjustments = max_threshold_adjustments
    
    def to_dict(self):
        return {
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'block_size': self.block_size,
            'stride': self.stride,
            'threshold': self.threshold,
            'dropout': self.dropout,
            'use_dynamic_threshold': self.use_dynamic_threshold,
            'max_threshold_adjustments': self.max_threshold_adjustments,
        }