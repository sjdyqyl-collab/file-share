import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

class XAttentionSimple(nn.Module):
    """
    Simplified XAttention implementation with basic antidiagonal scoring.
    
    Shape conventions:
    - B: batch size
    - L: sequence length
    - H: hidden size (d_model)
    - num_heads: number of attention heads
    - head_dim: dimension per head (H // num_heads)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int = 64,
        stride: int = 8,
        threshold: float = 0.9,
        dropout: float = 0.1
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # XAttention parameters
        self.block_size = block_size
        self.stride = stride
        self.threshold = threshold
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def compute_block_scores(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Compute importance scores for each block using simple antidiagonal approach.
        
        Args:
            Q: [B, num_heads, L, head_dim]
            K: [B, num_heads, L, head_dim]
            
        Returns:
            scores: [B, num_heads, num_blocks]
        """
        B, num_heads, L, head_dim = Q.shape
        num_blocks = (L + self.block_size - 1) // self.block_size
        
        scores = torch.zeros(B, num_heads, num_blocks, device=Q.device)
        
        for b in range(num_blocks):
            start = b * self.block_size
            end = min((b + 1) * self.block_size, L)
            
            # Extract blocks
            Q_block = Q[:, :, start:end, :]  # [B, num_heads, block_len, head_dim]
            K_block = K[:, :, start:end, :]  # [B, num_heads, block_len, head_dim]
            
            # Compute attention scores within block
            block_len = end - start
            if block_len > 0:
                # Simple scoring: average attention across block
                scores_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) / self.scale
                # scores_block: [B, num_heads, block_len, block_len]
                
                # Sum as importance score
                scores[:, :, b] = scores_block.sum(dim=(-2, -1)) / (block_len * block_len)
        
        return scores
    
    def select_blocks(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Select important blocks using threshold.
        
        Args:
            scores: [B, num_heads, num_blocks]
            
        Returns:
            block_masks: [B, num_heads, L, L]
        """
        B, num_heads, num_blocks = scores.shape
        L = num_blocks * self.block_size
        
        # Normalize scores
        scores_norm = F.softmax(scores, dim=-1)
        
        # Sort and select
        sorted_scores, sorted_indices = torch.sort(scores_norm, dim=-1, descending=True)
        cumulative_sum = torch.cumsum(sorted_scores, dim=-1)
        
        # Find cutoff
        cutoff_indices = torch.sum(cumulative_sum <= self.threshold, dim=-1)
        
        # Create masks
        block_masks = torch.zeros(B, num_heads, L, L, device=scores.device, dtype=torch.bool)
        
        for b in range(B):
            for h in range(num_heads):
                num_selected = min(cutoff_indices[b, h] + 1, num_blocks)
                selected_blocks = sorted_indices[b, h, :num_selected]
                
                for block_idx in selected_blocks:
                    start = block_idx * self.block_size
                    end = min((block_idx + 1) * self.block_size, L)
                    block_masks[b, h, :, start:end] = True
                    block_masks[b, h, start:end, :] = True
        
        return block_masks
    
    def sparse_attention(self, Q, K, V, block_masks, causal=False):
        """Compute sparse attention."""
        B, num_heads, L, head_dim = Q.shape
        output = torch.zeros_like(Q)
        
        for h in range(num_heads):
            mask = block_masks[:, h, :, :]
            
            Q_h = Q[:, h, :, :]
            K_h = K[:, h, :, :]
            V_h = V[:, h, :, :]
            
            scores = torch.bmm(Q_h, K_h.transpose(-2, -1)) / self.scale
            
            if causal:
                causal_mask = torch.triu(torch.ones(L, L, device=Q.device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(causal_mask, float('-inf'))
            
            scores = scores.masked_fill(~mask, float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output[:, h, :, :] = torch.bmm(attn_weights, V_h)
        
        return output
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: [B, L, H]
            key: [B, L, H]
            value: [B, L, H]
            
        Returns:
            output: [B, L, H]
            block_masks: [B, num_heads, L, L]
        """
        B, L, H = query.shape
        
        # Projections
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head
        Q = Q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scores and select blocks
        scores = self.compute_block_scores(Q, K)
        block_masks = self.select_blocks(scores)
        
        # Sparse attention
        sparse_out = self.sparse_attention(Q, K, V, block_masks, causal)
        
        # Reshape and project
        sparse_out = sparse_out.transpose(1, 2).contiguous().reshape(B, L, H)
        output = self.out_proj(sparse_out)
        
        return output, block_masks
    
    def load_weights(self, state_dict: dict):
        """Load pre-trained weights."""
        self.load_state_dict(state_dict)
    
    def set_threshold(self, threshold: float):
        """Set threshold."""
        self.threshold = threshold
    
    def get_sparsity_stats(self, block_masks: torch.Tensor) -> dict:
        """Get sparsity statistics."""
        total = block_masks.numel()
        selected = block_masks.sum().item()
        
        return {
            'sparsity': 1.0 - selected/total,
            'density': selected/total,
            'selected_blocks': selected,
            'total_blocks': total
        }