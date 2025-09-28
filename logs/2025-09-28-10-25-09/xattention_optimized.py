import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

class XAttentionOptimized(nn.Module):
    """
    Optimized XAttention implementation with vectorized antidiagonal scoring.
    
    Shape conventions:
    - B: batch size
    - L: sequence length
    - H: hidden size (d_model)
    - num_heads: number of attention heads
    - head_dim: dimension per head (H // num_heads)
    - B_block: block size for sparse attention
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int = 64,
        stride: int = 8,
        threshold: float = 0.9,
        dropout: float = 0.1,
        use_dynamic_threshold: bool = True
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
        self.use_dynamic_threshold = use_dynamic_threshold
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def compute_antidiagonal_scores_fast(
        self, Q: torch.Tensor, K: torch.Tensor
    ) -> torch.Tensor:
        """
        Fast antidiagonal scoring using efficient computation.
        
        Args:
            Q: [B, num_heads, L, head_dim]
            K: [B, num_heads, L, head_dim]
            
        Returns:
            scores: [B, num_heads, num_blocks]
        """
        B, num_heads, L, head_dim = Q.shape
        num_blocks = (L + self.block_size - 1) // self.block_size
        
        # Pad sequences to multiple of block_size
        pad_len = num_blocks * self.block_size - L
        if pad_len > 0:
            Q_pad = F.pad(Q, (0, 0, 0, pad_len))
            K_pad = F.pad(K, (0, 0, 0, pad_len))
        else:
            Q_pad = Q
            K_pad = K
        
        # Reshape to blocks: [B, num_heads, num_blocks, block_size, head_dim]
        Q_blocks = Q_pad.reshape(B, num_heads, num_blocks, self.block_size, head_dim)
        K_blocks = K_pad.reshape(B, num_heads, num_blocks, self.block_size, head_dim)
        
        # Initialize scores
        scores = torch.zeros(B, num_heads, num_blocks, device=Q.device)
        
        # Compute scores for each block
        for b in range(num_blocks):
            Q_block = Q_blocks[:, :, b, :, :]  # [B, num_heads, block_size, head_dim]
            K_block = K_blocks[:, :, b, :, :]  # [B, num_heads, block_size, head_dim]
            
            # Compute attention scores within block
            # Reshape for batch matrix multiplication
            Q_reshaped = Q_block.reshape(-1, self.block_size, head_dim)  # [B*num_heads, block_size, head_dim]
            K_reshaped = K_block.reshape(-1, self.block_size, head_dim)
            
            # Compute attention matrix
            attn_scores = torch.bmm(Q_reshaped, K_reshaped.transpose(-2, -1)) / self.scale
            # attn_scores: [B*num_heads, block_size, block_size]
            
            # Extract antidiagonal elements efficiently
            attn_scores = attn_scores.reshape(B, num_heads, self.block_size, self.block_size)
            
            # Create antidiagonal mask
            mask = self._create_antidiagonal_mask(self.block_size, self.stride, Q.device)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, block_size, block_size]
            
            # Apply mask and sum
            masked_scores = attn_scores * mask
            scores[:, :, b] = masked_scores.sum(dim=(-2, -1))
        
        return scores
    
    def _create_antidiagonal_mask(self, size: int, stride: int, device: torch.device) -> torch.Tensor:
        """Create antidiagonal mask for efficient computation."""
        mask = torch.zeros(size, size, device=device)
        
        for k in range(0, 2 * size - 1, stride):
            for i in range(max(0, k - size + 1), min(k + 1, size)):
                j = k - i
                if 0 <= j < size:
                    mask[i, j] = 1.0
        
        return mask
    
    def select_blocks_vectorized(
        self, scores: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        """
        Vectorized block selection using threshold.
        
        Args:
            scores: [B, num_heads, num_blocks]
            threshold: selection threshold
            
        Returns:
            block_masks: [B, num_heads, L, L]
        """
        B, num_heads, num_blocks = scores.shape
        L = num_blocks * self.block_size
        
        # Softmax normalization
        scores_norm = F.softmax(scores, dim=-1)  # [B, num_heads, num_blocks]
        
        # Sort scores and compute cumulative sum
        sorted_scores, sorted_indices = torch.sort(scores_norm, dim=-1, descending=True)
        cumulative_sum = torch.cumsum(sorted_scores, dim=-1)
        
        # Find cutoff indices
        cutoff_mask = cumulative_sum <= threshold
        cutoff_indices = cutoff_mask.sum(dim=-1)  # [B, num_heads]
        
        # Create block masks
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
    
    def sparse_attention_optimized(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        block_masks: torch.Tensor,
        causal: bool = False
    ) -> torch.Tensor:
        """
        Optimized sparse attention computation.
        
        Args:
            Q: [B, num_heads, L, head_dim]
            K: [B, num_heads, L, head_dim]
            V: [B, num_heads, L, head_dim]
            block_masks: [B, num_heads, L, L]
            causal: whether to apply causal masking
            
        Returns:
            output: [B, num_heads, L, head_dim]
        """
        B, num_heads, L, head_dim = Q.shape
        
        # Initialize output
        output = torch.zeros_like(Q)
        
        # Process each head separately for now (can be batched further)
        for h in range(num_heads):
            mask = block_masks[:, h, :, :]  # [B, L, L]
            
            Q_h = Q[:, h, :, :]  # [B, L, head_dim]
            K_h = K[:, h, :, :]  # [B, L, head_dim]
            V_h = V[:, h, :, :]  # [B, L, head_dim]
            
            # Compute attention scores
            scores = torch.bmm(Q_h, K_h.transpose(-2, -1)) / self.scale  # [B, L, L]
            
            # Apply causal mask if needed
            if causal:
                causal_mask = torch.triu(torch.ones(L, L, device=Q.device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(causal_mask, float('-inf'))
            
            # Apply block mask
            scores = scores.masked_fill(~mask, float('-inf'))
            
            # Softmax and apply attention
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            out_h = torch.bmm(attn_weights, V_h)  # [B, L, head_dim]
            output[:, h, :, :] = out_h
        
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
        Optimized XAttention forward pass.
        
        Args:
            query: [B, L, H]
            key: [B, L, H]
            value: [B, L, H]
            attention_mask: [B, L, L] or None
            causal: whether to apply causal masking
            
        Returns:
            output: [B, L, H]
            block_masks: [B, num_heads, L, L]
        """
        B, L, H = query.shape
        
        # Linear projections
        Q = self.q_proj(query)  # [B, L, H]
        K = self.k_proj(key)    # [B, L, H]
        V = self.v_proj(value)  # [B, L, H]
        
        # Reshape for multi-head attention
        Q = Q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        K = K.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        V = V.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        
        # Compute antidiagonal scores
        scores = self.compute_antidiagonal_scores_fast(Q, K)  # [B, num_heads, num_blocks]
        
        # Select blocks
        block_masks = self.select_blocks_vectorized(scores, self.threshold)
        
        # Compute sparse attention
        sparse_out = self.sparse_attention_optimized(Q, K, V, block_masks, causal)
        
        # Reshape and project output
        sparse_out = sparse_out.transpose(1, 2).contiguous().reshape(B, L, H)  # [B, L, H]
        output = self.out_proj(sparse_out)
        
        return output, block_masks
    
    def load_weights(self, state_dict: dict):
        """Load pre-trained weights if available."""
        self.load_state_dict(state_dict)
    
    def set_threshold(self, threshold: float):
        """Set global threshold for all heads."""
        self.threshold = threshold
    
    def get_sparsity_stats(self, block_masks: torch.Tensor) -> dict:
        """Compute sparsity statistics from block masks."""
        total_elements = block_masks.numel()
        selected_elements = block_masks.sum().item()
        
        sparsity = 1.0 - (selected_elements / total_elements)
        density = selected_elements / total_elements
        
        return {
            'sparsity': sparsity,
            'density': density,
            'selected_blocks': selected_elements,
            'total_blocks': total_elements
        }