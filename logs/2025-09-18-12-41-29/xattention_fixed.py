"""
Fixed XAttention: Block Sparse Attention with Antidiagonal Scoring
Fixed searchsorted usage and other issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class XAttentionFixed(nn.Module):
    """
    Fixed XAttention implementation with antidiagonal scoring for block-sparse attention
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        block_size: int = 8,
        stride: int = 8,
        threshold: float = 0.9,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        causal: bool = True,
        use_dynamic_threshold: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0
        assert block_size % stride == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        
        # XAttention specific parameters
        self.block_size = block_size
        self.stride = stride
        self.threshold = threshold
        self.use_dynamic_threshold = use_dynamic_threshold
        
        # Initialize per-head thresholds
        self.register_buffer('head_thresholds', torch.ones(num_heads) * threshold)
        
        # Linear projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Cache for selected blocks
        self.selected_blocks_cache = None
        
    def compute_antidiagonal_score(
        self, 
        q_block: torch.Tensor, 
        k_block: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute antidiagonal score for a block of attention
        """
        B, H, B_size, d_h = q_block.shape
        
        # Compute attention scores for the block
        scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale
        
        # Initialize score accumulator
        antidiagonal_sum = torch.zeros(B, H, device=q_block.device)
        
        # Compute strided antidiagonal sums
        for offset in range(0, B_size, self.stride):
            # Create mask for antidiagonal pattern
            mask = torch.zeros(B_size, B_size, device=q_block.device)
            
            # Fill antidiagonal with stride
            for i in range(B_size):
                j = (B_size - 1 - i + offset) % B_size
                mask[i, j] = 1.0
            
            # Apply mask and sum
            masked_scores = scores * mask.unsqueeze(0).unsqueeze(0)
            antidiagonal_sum += masked_scores.sum(dim=(-2, -1))
        
        return antidiagonal_sum
    
    def select_blocks(
        self, 
        scores: torch.Tensor,
        threshold: float
    ) -> List[List[int]]:
        """
        Select blocks based on cumulative probability threshold
        """
        B, H, num_blocks = scores.shape
        selected_blocks = []
        
        for b in range(B):
            batch_blocks = []
            for h in range(H):
                # Get scores for this head
                head_scores = scores[b, h]
                
                # Convert to probabilities
                probs = F.softmax(head_scores, dim=-1)
                
                # Sort by probability in descending order
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                
                # Find minimal set for cumulative probability >= threshold
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Fix: searchsorted needs scalar input for 1D boundaries
                # Find first index where cumulative probability >= threshold
                cutoff_idx = 0
                for i, cum_prob in enumerate(cumulative_probs):
                    if cum_prob >= threshold:
                        cutoff_idx = i
                        break
                else:
                    cutoff_idx = len(cumulative_probs) - 1
                
                # Get selected indices
                selected = sorted_indices[:cutoff_idx + 1].tolist()
                batch_blocks.append(selected)
            
            selected_blocks.append(batch_blocks)
        
        return selected_blocks
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with XAttention block-sparse attention
        """
        B, L, D = x.shape
        
        # Ensure sequence length is divisible by block size
        if L % self.block_size != 0:
            pad_len = self.block_size - (L % self.block_size)
            x = F.pad(x, (0, 0, 0, pad_len))
            L_padded = L + pad_len
        else:
            L_padded = L
        
        num_blocks = L_padded // self.block_size
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, L_padded, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Reshape for block processing
        q_blocks = q.reshape(B, self.num_heads, num_blocks, self.block_size, self.head_dim)
        k_blocks = k.reshape(B, self.num_heads, num_blocks, self.block_size, self.head_dim)
        v_blocks = v.reshape(B, self.num_heads, num_blocks, self.block_size, self.head_dim)
        
        # Compute block scores using antidiagonal scoring
        block_scores = torch.zeros(B, self.num_heads, num_blocks, device=x.device)
        
        for b_idx in range(num_blocks):
            q_block = q_blocks[:, :, b_idx]
            k_block = k_blocks[:, :, b_idx]
            
            # Compute antidiagonal score for this block
            score = self.compute_antidiagonal_score(q_block, k_block)
            block_scores[:, :, b_idx] = score
        
        # Select blocks based on thresholds
        selected_blocks = []
        for b in range(B):
            batch_blocks = []
            for h in range(self.num_heads):
                threshold = self.head_thresholds[h]
                head_scores = block_scores[b, h]
                
                # Convert to probabilities
                probs = F.softmax(head_scores, dim=-1)
                
                # Select blocks
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Fix: searchsorted issue
                cutoff_idx = 0
                for i, cum_prob in enumerate(cumulative_probs):
                    if cum_prob >= threshold:
                        cutoff_idx = i
                        break
                else:
                    cutoff_idx = len(cumulative_probs) - 1
                
                selected = sorted_indices[:cutoff_idx + 1].tolist()
                batch_blocks.append(selected)
            
            selected_blocks.append(batch_blocks)
        
        # Cache selected blocks for analysis
        self.selected_blocks_cache = selected_blocks
        
        # Compute sparse attention
        out = torch.zeros_like(q)
        
        for b in range(B):
            for h in range(self.num_heads):
                selected = selected_blocks[b][h]
                
                if not selected:
                    continue
                
                # Gather selected blocks
                q_selected = []
                k_selected = []
                v_selected = []
                
                for block_idx in selected:
                    start = block_idx * self.block_size
                    end = start + self.block_size
                    q_selected.append(q[b, h, start:end])
                    k_selected.append(k[b, h, start:end])
                    v_selected.append(v[b, h, start:end])
                
                if q_selected:
                    q_flat = torch.cat(q_selected, dim=0)
                    k_flat = torch.cat(k_selected, dim=0)
                    v_flat = torch.cat(v_selected, dim=0)
                    
                    # Compute attention scores
                    scores = torch.matmul(q_flat, k_flat.t()) * self.scale
                    
                    # Apply causal mask if needed
                    if self.causal:
                        total_len = len(q_flat)
                        causal_mask = torch.triu(
                            torch.ones(total_len, total_len, device=x.device, dtype=torch.bool),
                            diagonal=1
                        )
                        scores = scores.masked_fill(causal_mask, float('-inf'))
                    
                    attn = F.softmax(scores, dim=-1)
                    attn = self.attn_drop(attn)
                    
                    out_flat = torch.matmul(attn, v_flat)
                    
                    # Scatter results back to output
                    idx = 0
                    for block_idx in selected:
                        start = block_idx * self.block_size
                        end = start + self.block_size
                        out[b, h, start:end] = out_flat[idx:idx + self.block_size]
                        idx += self.block_size
        
        # Reshape output
        out = out.transpose(1, 2).reshape(B, L_padded, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # Remove padding if added
        if L_padded != L:
            out = out[:, :L, :]
        
        return out
    
    def get_sparsity_stats(self) -> dict:
        """Get sparsity statistics from the last forward pass"""
        if self.selected_blocks_cache is None:
            return {}
        
        total_blocks = 0
        selected_blocks = 0
        
        for batch_blocks in self.selected_blocks_cache:
            for head_blocks in batch_blocks:
                total_blocks += len(head_blocks)
                selected_blocks += len(head_blocks)
        
        density = selected_blocks / total_blocks if total_blocks > 0 else 0.0
        
        return {
            'density': density,
            'total_blocks': total_blocks,
            'selected_blocks': selected_blocks,
            'sparsity': 1.0 - density
        }
    
    def load_weights(self, state_dict: dict):
        """Load weights from state dictionary"""
        self.load_state_dict(state_dict)
        
    def save_weights(self) -> dict:
        """Save weights to state dictionary"""
        return self.state_dict()


class XAttentionOptimizedFixed(nn.Module):
    """
    Fixed Optimized XAttention implementation with batched operations
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        block_size: int = 8,
        stride: int = 8,
        threshold: float = 0.9,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        causal: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        
        self.block_size = block_size
        self.stride = stride
        self.threshold = threshold
        
        # Linear projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def batched_antidiagonal_score(
        self,
        q: torch.Tensor,
        k: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute antidiagonal scores for all blocks in batch
        """
        B, H, L, d_h = q.shape
        num_blocks = L // self.block_size
        
        # Reshape to blocks
        q_blocks = q.reshape(B, H, num_blocks, self.block_size, d_h)
        k_blocks = k.reshape(B, H, num_blocks, self.block_size, d_h)
        
        # Compute attention scores for all blocks
        scores = torch.matmul(
            q_blocks, k_blocks.transpose(-2, -1)
        ) * self.scale
        
        # Create antidiagonal mask
        B_size = self.block_size
        mask = torch.zeros(B_size, B_size, device=q.device)
        
        for offset in range(0, B_size, self.stride):
            for i in range(B_size):
                j = (B_size - 1 - i + offset) % B_size
                mask[i, j] = 1.0
        
        # Apply mask and sum
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        masked_scores = scores * mask
        block_scores = masked_scores.sum(dim=(-2, -1))
        
        return block_scores
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Optimized forward pass with batched operations
        """
        B, L, D = x.shape
        
        # Pad to block size
        if L % self.block_size != 0:
            pad_len = self.block_size - (L % self.block_size)
            x = F.pad(x, (0, 0, 0, pad_len))
            L_padded = L + pad_len
        else:
            L_padded = L
        
        num_blocks = L_padded // self.block_size
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, L_padded, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute antidiagonal scores
        block_scores = self.batched_antidiagonal_score(q, k)
        
        # Select blocks based on threshold
        probs = F.softmax(block_scores, dim=-1)
        
        # Compute cumulative probabilities
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Fix: searchsorted issue - find cutoff indices manually
        B, H, num_blocks = probs.shape
        cutoff_indices = torch.zeros(B, H, dtype=torch.long, device=x.device)
        
        for b in range(B):
            for h in range(H):
                for i in range(num_blocks):
                    if cumulative_probs[b, h, i] >= self.threshold:
                        cutoff_indices[b, h] = i
                        break
                else:
                    cutoff_indices[b, h] = num_blocks - 1
        
        # Create selection mask
        selection_mask = torch.zeros_like(probs, dtype=torch.bool)
        for b in range(B):
            for h in range(self.num_heads):
                cutoff = cutoff_indices[b, h].item()
                selected = sorted_indices[b, h, :cutoff + 1]
                selection_mask[b, h, selected] = True
        
        # Compute sparse attention
        out = torch.zeros_like(q)
        
        for b in range(B):
            for h in range(self.num_heads):
                selected_blocks = torch.where(selection_mask[b, h])[0]
                
                if len(selected_blocks) == 0:
                    continue
                
                # Gather selected blocks
                start_indices = selected_blocks * self.block_size
                end_indices = start_indices + self.block_size
                
                # Extract selected regions
                q_selected = []
                k_selected = []
                v_selected = []
                
                for start, end in zip(start_indices, end_indices):
                    q_selected.append(q[b, h, start:end])
                    k_selected.append(k[b, h, start:end])
                    v_selected.append(v[b, h, start:end])
                
                if q_selected:
                    q_flat = torch.cat(q_selected, dim=0)
                    k_flat = torch.cat(k_selected, dim=0)
                    v_flat = torch.cat(v_selected, dim=0)
                    
                    # Compute attention
                    scores = torch.matmul(q_flat, k_flat.t()) * self.scale
                    
                    if self.causal:
                        total_len = len(q_flat)
                        causal_mask = torch.triu(
                            torch.ones(total_len, total_len, device=x.device, dtype=torch.bool),
                            diagonal=1
                        )
                        scores = scores.masked_fill(causal_mask, float('-inf'))
                    
                    attn = F.softmax(scores, dim=-1)
                    attn = self.attn_drop(attn)
                    out_flat = torch.matmul(attn, v_flat)
                    
                    # Scatter results back
                    idx = 0
                    for start, end in zip(start_indices, end_indices):
                        block_size = end - start
                        out[b, h, start:end] = out_flat[idx:idx + block_size]
                        idx += block_size
        
        # Final projection
        out = out.transpose(1, 2).reshape(B, L_padded, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # Remove padding
        if L_padded != L:
            out = out[:, :L, :]
        
        return out
    
    def load_weights(self, state_dict: dict):
        """Load weights from state dictionary"""
        self.load_state_dict(state_dict)
        
    def save_weights(self) -> dict:
        """Save weights to state dictionary"""
        return self.state_dict()