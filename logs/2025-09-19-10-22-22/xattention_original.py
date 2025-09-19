"""
Original XAttention implementation as described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any


class XAttentionOriginal(nn.Module):
    """
    Original XAttention implementation with antidiagonal scoring and threshold-based block selection.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        block_size: int = 8,
        stride: int = 8,
        threshold: float = 0.9,
        max_seq_len: int = 8192,
        use_dynamic_threshold: bool = False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block_size = block_size
        self.stride = stride
        self.threshold = threshold
        self.max_seq_len = max_seq_len
        self.use_dynamic_threshold = use_dynamic_threshold
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Dynamic threshold parameters
        if use_dynamic_threshold:
            self.threshold_params = nn.Parameter(torch.ones(num_heads) * threshold)
            self.register_buffer('threshold_history', torch.zeros(num_heads, 1000))
            self.register_buffer('threshold_ptr', torch.zeros(1, dtype=torch.long))
        
    def _compute_antidiagonal_scores(self, attn_block: torch.Tensor) -> torch.Tensor:
        """
        Compute antidiagonal scores for a given attention block.
        
        Args:
            attn_block: [B, B] attention values for a block
            
        Returns:
            score: scalar importance score for the block
        """
        B = attn_block.size(0)
        scores = []
        
        # Sample along antidiagonals with given stride
        for k in range(0, 2 * B - 1, self.stride):
            # Get elements on the k-th antidiagonal
            elements = []
            for i in range(B):
                j = k - i
                if 0 <= j < B:
                    elements.append(attn_block[i, j])
            
            if elements:
                scores.append(torch.stack(elements).sum())
        
        if not scores:
            return torch.tensor(0.0, device=attn_block.device)
        
        return torch.stack(scores).sum()
    
    def _select_blocks(self, attn_approx: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Select blocks based on importance scores and threshold.
        
        Args:
            attn_approx: [NB, B, B] approximate attention map for all blocks
            threshold: selection threshold
            
        Returns:
            mask: [L, L] binary mask indicating selected blocks
        """
        L = attn_approx.size(0) * self.block_size
        NB = attn_approx.size(0)
        
        # Compute scores for each block
        scores = []
        for b in range(NB):
            score = self._compute_antidiagonal_scores(attn_approx[b])
            scores.append(score)
        
        scores = torch.stack(scores)
        
        # Normalize scores
        scores = F.softmax(scores, dim=0)
        
        # Select blocks until cumulative sum exceeds threshold
        sorted_indices = torch.argsort(scores, descending=True)
        cumulative_sum = torch.cumsum(scores[sorted_indices], dim=0)
        
        selected_blocks = sorted_indices[cumulative_sum <= threshold]
        if len(selected_blocks) == 0:
            selected_blocks = sorted_indices[:max(1, int(threshold * NB))]
        
        # Create mask
        mask = torch.zeros(L, L, dtype=torch.bool, device=attn_approx.device)
        for b in selected_blocks:
            start = b * self.block_size
            end = (b + 1) * self.block_size
            mask[start:end, start:end] = True
        
        return mask
    
    def _approximate_attention_map(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Create approximate attention map using strided antidiagonal sampling.
        
        Args:
            q: [L, d] query matrix
            k: [L, d] key matrix
            
        Returns:
            attn_approx: [NB, B, B] approximate attention map
        """
        L, d = q.size()
        NB = L // self.block_size
        
        attn_approx = []
        
        for b in range(NB):
            q_slice = q[b * self.block_size:(b + 1) * self.block_size]
            k_slice = k[b * self.block_size:(b + 1) * self.block_size]
            
            # Reshape along antidiagonal for strided computation
            q_reshaped = self._reshape_along_antidiagonal(q_slice)
            k_reshaped = self._reshape_along_antidiagonal(k_slice)
            
            # Compute approximate attention
            scale = 1.0 / np.sqrt(self.head_dim * self.stride)
            attn = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            
            # Average back to block size
            attn_block = attn.mean(dim=0)  # [B, B]
            attn_approx.append(attn_block)
        
        return torch.stack(attn_approx)
    
    def _reshape_along_antidiagonal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape tensor along antidiagonal for strided computation.
        
        Args:
            x: [B, d] input tensor
            
        Returns:
            reshaped: [S, B, d//S] reshaped tensor
        """
        B, d = x.size()
        if d % self.stride != 0:
            # Pad if necessary
            pad_size = self.stride - (d % self.stride)
            x = F.pad(x, (0, pad_size))
            d = x.size(1)
        
        return x.view(B, self.stride, d // self.stride).transpose(0, 1)
    
    def _update_threshold(self, head_idx: int, performance: float):
        """
        Update threshold using dynamic programming approach.
        
        Args:
            head_idx: index of attention head
            performance: current performance metric
        """
        if not self.use_dynamic_threshold:
            return
        
        # Simple threshold adjustment based on performance
        ptr = int(self.threshold_ptr.item())
        self.threshold_history[head_idx, ptr] = performance
        
        if ptr > 0 and self.threshold_history[head_idx, ptr] < self.threshold_history[head_idx, ptr-1]:
            # Decrease threshold if performance drops
            self.threshold_params[head_idx] *= 0.9
        
        self.threshold_ptr[0] = (ptr + 1) % 1000
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of XAttention.
        
        Args:
            x: [B, L, D] input tensor
            mask: [B, L] attention mask (optional)
            return_attention: whether to return attention weights
            
        Returns:
            dict with output and optionally attention weights
        """
        B, L, D = x.shape
        
        # Linear projections
        q = self.q_proj(x)  # [B, L, D]
        k = self.k_proj(x)  # [B, L, D]
        v = self.v_proj(x)  # [B, L, D]
        
        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        
        outputs = []
        attention_weights = []
        
        for h in range(self.num_heads):
            # Get head-specific parameters
            if self.use_dynamic_threshold:
                threshold = torch.sigmoid(self.threshold_params[h]).item()
            else:
                threshold = self.threshold
            
            # Process each head
            q_h = q[:, h]  # [B, L, d]
            k_h = k[:, h]  # [B, L, d]
            v_h = v[:, h]  # [B, L, d]
            
            # Handle batch dimension
            batch_outputs = []
            batch_attention = []
            
            for b in range(B):
                q_b = q_h[b]  # [L, d]
                k_b = k_h[b]  # [L, d]
                v_b = v_h[b]  # [L, d]
                
                # Pad sequence length to be divisible by block_size
                pad_len = (self.block_size - L % self.block_size) % self.block_size
                if pad_len > 0:
                    q_b = F.pad(q_b, (0, 0, 0, pad_len))
                    k_b = F.pad(k_b, (0, 0, 0, pad_len))
                    v_b = F.pad(v_b, (0, 0, 0, pad_len))
                
                padded_len = q_b.size(0)
                
                # Approximate attention map for block selection
                attn_approx = self._approximate_attention_map(q_b, k_b)
                
                # Select important blocks
                mask_blocks = self._select_blocks(attn_approx, threshold)
                mask_blocks = mask_blocks[:L, :L]  # Remove padding
                
                # Compute sparse attention
                scale = 1.0 / np.sqrt(self.head_dim)
                
                # Create sparse attention mask
                attn_mask = torch.zeros(L, L, device=x.device)
                attn_mask[mask_blocks] = 1.0
                
                if mask is not None:
                    # Apply input mask
                    attn_mask = attn_mask * mask[b].unsqueeze(0) * mask[b].unsqueeze(1)
                
                # Compute attention
                attn_scores = torch.matmul(q_b[:L], k_b[:L].transpose(-2, -1)) * scale
                attn_scores = attn_scores.masked_fill(~mask_blocks[:L, :L], float('-inf'))
                attn_weights_h = F.softmax(attn_scores, dim=-1)
                
                # Apply attention to values
                out_b = torch.matmul(attn_weights_h, v_b[:L])
                
                batch_outputs.append(out_b)
                batch_attention.append(attn_weights_h)
                
                # Update threshold if using dynamic adjustment
                if self.use_dynamic_threshold:
                    # Use attention entropy as performance metric
                    entropy = -(attn_weights_h * attn_weights_h.clamp(min=1e-8).log()).sum(-1).mean()
                    self._update_threshold(h, entropy.item())
            
            outputs.append(torch.stack(batch_outputs).transpose(0, 1))  # [B, L, d]
            attention_weights.append(torch.stack(batch_attention))  # [B, L, L]
        
        # Concatenate heads
        out = torch.cat(outputs, dim=-1)  # [B, L, D]
        out = self.out_proj(out)
        
        result = {'output': out}
        if return_attention:
            result['attention_weights'] = torch.stack(attention_weights, dim=1)  # [B, H, L, L]
        
        return result
    
    def load_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Load pre-trained weights."""
        self.load_state_dict(state_dict)
    
    def get_sparsity_stats(self) -> Dict[str, float]:
        """Get sparsity statistics."""
        return {
            'block_size': self.block_size,
            'stride': self.stride,
            'threshold': self.threshold,
            'use_dynamic_threshold': self.use_dynamic_threshold
        }