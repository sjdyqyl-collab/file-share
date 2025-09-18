"""
Original XAttention Implementation
This implements the XAttention paper's method with antidiagonal scoring for block sparse attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class XAttentionOriginal(nn.Module):
    """
    Original XAttention implementation as described in the paper.
    Uses antidiagonal scoring for block sparse attention with dynamic threshold prediction.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, block_size: int = 8, 
                 stride: int = 8, dropout: float = 0.0, max_threshold_adjustments: int = 1000):
        """
        Initialize XAttention.
        
        Args:
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
            block_size: Size of attention blocks (B×B)
            stride: Stride parameter for antidiagonal sampling
            dropout: Dropout probability
            max_threshold_adjustments: Maximum threshold adjustments for DP
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size
        self.stride = stride
        self.max_threshold_adjustments = max_threshold_adjustments
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        
        # Thresholds for each head (initialized to 0.9 as in paper)
        self.register_buffer('thresholds', torch.ones(num_heads) * 0.9)
        
    def compute_antidiagonal_scores(self, attention_block: torch.Tensor) -> torch.Tensor:
        """
        Compute antidiagonal scores for a block of attention weights.
        
        Args:
            attention_block: Attention block of shape [B, B]
            
        Returns:
            Antidiagonal score
        """
        B = attention_block.shape[0]
        scores = []
        
        # Iterate over antidiagonals
        for k in range(2 * B - 1):
            # Get indices for this antidiagonal
            i_indices = []
            j_indices = []
            
            for i in range(max(0, k - B + 1), min(k + 1, B)):
                j = k - i
                if 0 <= j < B and (i + j) % self.stride == 0:
                    i_indices.append(i)
                    j_indices.append(j)
            
            if i_indices:
                # Sum values along this strided antidiagonal
                score = attention_block[i_indices, j_indices].sum()
                scores.append(score)
        
        return torch.tensor(scores).sum() if scores else torch.tensor(0.0)
    
    def compute_block_importance(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Compute importance scores for all blocks using antidiagonal scoring.
        
        Args:
            q: Query tensor [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            Importance scores [batch_size, num_heads, num_blocks]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Calculate number of blocks
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        # Initialize importance scores
        importance_scores = torch.zeros(batch_size, num_heads, num_blocks, num_blocks)
        
        # Compute attention scores for each block
        for b_i in range(num_blocks):
            for b_j in range(num_blocks):
                # Get block boundaries
                i_start = b_i * self.block_size
                i_end = min((b_i + 1) * self.block_size, seq_len)
                j_start = b_j * self.block_size
                j_end = min((b_j + 1) * self.block_size, seq_len)
                
                if i_start < seq_len and j_start < seq_len:
                    # Extract Q and K for this block
                    q_block = q[:, :, i_start:i_end, :]
                    k_block = k[:, :, j_start:j_end, :]
                    
                    # Compute attention for this block
                    attn_block = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale
                    attn_block = F.softmax(attn_block, dim=-1)
                    
                    # Compute antidiagonal score for each head
                    for head in range(num_heads):
                        score = self.compute_antidiagonal_scores(attn_block[0, head])
                        importance_scores[0, head, b_i, b_j] = score
        
        return importance_scores
    
    def select_blocks(self, importance_scores: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Select blocks based on importance scores and threshold.
        
        Args:
            importance_scores: Importance scores [batch_size, num_heads, num_blocks, num_blocks]
            threshold: Selection threshold
            
        Returns:
            Block mask [batch_size, num_heads, num_blocks, num_blocks]
        """
        batch_size, num_heads, num_blocks, _ = importance_scores.shape
        
        # Flatten scores for each head
        block_masks = torch.zeros_like(importance_scores, dtype=torch.bool)
        
        for b in range(batch_size):
            for h in range(num_heads):
                # Flatten the 2D importance scores
                flat_scores = importance_scores[b, h].flatten()
                
                if flat_scores.sum() > 0:
                    # Normalize to probabilities
                    probs = F.softmax(flat_scores, dim=0)
                    
                    # Sort in descending order
                    sorted_probs, indices = torch.sort(probs, descending=True)
                    
                    # Select blocks until cumulative probability >= threshold
                    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                    selected_indices = indices[cumulative_probs < threshold]
                    
                    # Handle edge case where threshold is very low
                    if len(selected_indices) == 0 and len(indices) > 0:
                        selected_indices = indices[:1]
                    
                    # Convert flat indices back to 2D
                    for idx in selected_indices:
                        i = idx // num_blocks
                        j = idx % num_blocks
                        block_masks[b, h, i, j] = True
        
        return block_masks
    
    def dynamic_threshold_prediction(self, importance_scores: torch.Tensor, 
                                   performance_scores: torch.Tensor) -> torch.Tensor:
        """
        Use dynamic programming to optimize thresholds per head.
        
        Args:
            importance_scores: Importance scores for all heads
            performance_scores: Performance scores for different thresholds
            
        Returns:
            Optimized thresholds for each head
        """
        # This is a simplified version - in practice, this would use actual performance metrics
        # For now, we'll use a heuristic based on score distribution
        
        batch_size, num_heads, num_blocks, _ = importance_scores.shape
        new_thresholds = torch.ones(num_heads)
        
        for h in range(num_heads):
            # Compute statistics for this head
            scores = importance_scores[:, h].flatten()
            if scores.std() > 0:
                # Adjust threshold based on score distribution
                # Higher std -> more selective threshold
                new_thresholds[h] = max(0.7, min(0.95, 0.9 - 0.1 * (scores.std() / scores.mean())))
            else:
                new_thresholds[h] = 0.9
        
        return new_thresholds
    
    def sparse_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        block_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute sparse attention using selected blocks.
        
        Args:
            q: Query tensor [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor [batch_size, num_heads, seq_len, head_dim]
            v: Value tensor [batch_size, num_heads, seq_len, head_dim]
            block_mask: Block selection mask [batch_size, num_heads, num_blocks, num_blocks]
            
        Returns:
            Attention output [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        num_blocks = block_mask.shape[-1]
        
        # Initialize output
        output = torch.zeros_like(q)
        
        # Compute attention only on selected blocks
        for b in range(batch_size):
            for h in range(num_heads):
                for i in range(num_blocks):
                    for j in range(num_blocks):
                        if block_mask[b, h, i, j]:
                            # Get block boundaries
                            i_start = i * self.block_size
                            i_end = min((i + 1) * self.block_size, seq_len)
                            j_start = j * self.block_size
                            j_end = min((j + 1) * self.block_size, seq_len)
                            
                            # Extract block tensors
                            q_block = q[b:b+1, h:h+1, i_start:i_end, :]
                            k_block = k[b:b+1, h:h+1, j_start:j_end, :]
                            v_block = v[b:b+1, h:h+1, j_start:j_end, :]
                            
                            # Compute attention for this block
                            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale
                            attn_weights = F.softmax(scores, dim=-1)
                            attn_weights = self.dropout(attn_weights)
                            
                            # Apply to values
                            attn_output = torch.matmul(attn_weights, v_block)
                            
                            # Add to output
                            output[b:b+1, h:h+1, i_start:i_end, :] += attn_output[0]
        
        return output
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of XAttention.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute block importance scores
        importance_scores = self.compute_block_importance(q, k)
        
        # Select blocks using current thresholds
        block_masks = []
        for h in range(self.num_heads):
            head_scores = importance_scores[:, h:h+1]
            threshold = self.thresholds[h]
            head_mask = self.select_blocks(head_scores, threshold)
            block_masks.append(head_mask)
        
        block_mask = torch.cat(block_masks, dim=1)
        
        # Compute sparse attention
        attn_output = self.sparse_attention(q, k, v, block_mask)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)
        
        return output
    
    def get_sparsity_info(self, x: torch.Tensor) -> dict:
        """
        Get sparsity information for analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with sparsity metrics
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute importance scores
        importance_scores = self.compute_block_importance(q, k)
        
        # Select blocks
        block_masks = []
        for h in range(self.num_heads):
            head_scores = importance_scores[:, h:h+1]
            threshold = self.thresholds[h]
            head_mask = self.select_blocks(head_scores, threshold)
            block_masks.append(head_mask)
        
        block_mask = torch.cat(block_masks, dim=1)
        
        # Calculate sparsity
        total_blocks = block_mask.numel()
        selected_blocks = block_mask.sum().item()
        density = selected_blocks / total_blocks
        sparsity = 1 - density
        
        return {
            'total_blocks': total_blocks,
            'selected_blocks': selected_blocks,
            'density': density,
            'sparsity': sparsity,
            'thresholds': self.thresholds.clone()
        }
    
    def get_flops(self, seq_len: int) -> int:
        """
        Calculate FLOPs for XAttention computation.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Number of FLOPs (approximate)
        """
        # Pattern selection: O(L²/S)
        pattern_flops = seq_len * seq_len // self.stride
        
        # Sparse attention: O(L²·ρ) where ρ is density (assume ~0.2 average)
        avg_density = 0.2
        sparse_flops = int(seq_len * seq_len * avg_density * self.hidden_size)
        
        return pattern_flops + sparse_flops


def test_xattention_original():
    """Test the original XAttention implementation."""
    torch.manual_seed(42)
    
    batch_size, seq_len, hidden_size, num_heads = 1, 64, 256, 8
    
    # Create attention module
    attention = XAttentionOriginal(hidden_size, num_heads, block_size=8, stride=8)
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass
    output = attention(x)
    
    # Get sparsity info
    sparsity_info = attention.get_sparsity_info(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sparsity info: {sparsity_info}")
    print(f"FLOPs for sequence length {seq_len}: {attention.get_flops(seq_len):,}")


if __name__ == "__main__":
    test_xattention_original()