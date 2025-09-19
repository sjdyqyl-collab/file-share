"""
XAttention: Block Sparse Attention with Antidiagonal Scoring
Original implementation as described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any


class XAttentionBase(nn.Module):
    """
    Original XAttention implementation using antidiagonal scoring for block-sparse attention.
    
    This class implements the core XAttention method that uses antidiagonal values
    as a proxy for block importance in attention matrices.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int = 16,
        stride: int = 8,
        threshold: float = 0.9,
        use_dynamic_threshold: bool = True,
        max_seq_length: int = 256000,
    ):
        """
        Initialize XAttention.
        
        Args:
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
            block_size: Size of each attention block (B×B)
            stride: Stride for antidiagonal selection
            threshold: Importance threshold for block selection
            use_dynamic_threshold: Whether to use dynamic threshold prediction
            max_seq_length: Maximum sequence length for initialization
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size
        self.stride = stride
        self.threshold = threshold
        self.use_dynamic_threshold = use_dynamic_threshold
        self.max_seq_length = max_seq_length
        
        # Validate dimensions
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Initialize dynamic threshold parameters if needed
        if use_dynamic_threshold:
            self.threshold_predictor = nn.Parameter(torch.ones(num_heads) * threshold)
            self.register_buffer('threshold_decay', torch.tensor(0.9))
            
        # Cache for computed patterns
        self.register_buffer('cached_patterns', None)
        self.register_buffer('cached_seq_len', torch.tensor(-1))
        
    def reshape_antidiagonal(self, x: torch.Tensor, stride: int) -> torch.Tensor:
        """
        Reshape tensor along antidiagonals with given stride.
        
        Args:
            x: Input tensor of shape [batch, seq_len, head_dim]
            stride: Stride for antidiagonal selection
            
        Returns:
            Reshaped tensor along antidiagonals
        """
        batch, seq_len, head_dim = x.shape
        
        # Create antidiagonal indices
        indices = []
        for offset in range(-seq_len + 1, seq_len):
            diag_indices = []
            for i in range(max(0, offset), min(seq_len, seq_len + offset)):
                j = i - offset
                if 0 <= j < seq_len and (i + j) % stride == 0:
                    diag_indices.append(i)
            if diag_indices:
                indices.extend(diag_indices)
        
        if not indices:
            indices = list(range(0, seq_len, stride))
            
        # Select antidiagonal elements
        x_reshaped = x[:, indices, :]
        return x_reshaped
    
    def compute_block_importance(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        block_size: int,
        stride: int,
    ) -> torch.Tensor:
        """
        Compute importance scores for each block using antidiagonal values.
        
        Args:
            q: Query tensor [batch, seq_len, head_dim]
            k: Key tensor [batch, seq_len, head_dim]
            block_size: Size of each block
            stride: Stride for antidiagonal selection
            
        Returns:
            Block importance scores [batch, num_blocks, num_blocks]
        """
        batch, seq_len, head_dim = q.shape
        num_blocks = (seq_len + block_size - 1) // block_size
        
        # Initialize importance matrix
        importance = torch.zeros(batch, num_blocks, num_blocks, device=q.device)
        
        for b_i in range(num_blocks):
            for b_j in range(num_blocks):
                # Extract block
                start_i, end_i = b_i * block_size, min((b_i + 1) * block_size, seq_len)
                start_j, end_j = b_j * block_size, min((b_j + 1) * block_size, seq_len)
                
                q_block = q[:, start_i:end_i, :]
                k_block = k[:, start_j:end_j, :]
                
                # Reshape along antidiagonals
                q_reshaped = self.reshape_antidiagonal(q_block, stride)
                k_reshaped = self.reshape_antidiagonal(k_block, stride)
                
                if q_reshaped.shape[1] > 0 and k_reshaped.shape[1] > 0:
                    # Compute approximate attention
                    scores = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1))
                    scores = scores / np.sqrt(head_dim * stride)
                    attn_weights = F.softmax(scores, dim=-1)
                    
                    # Sum of antidiagonal values as importance
                    importance[:, b_i, b_j] = attn_weights.sum(dim=(-2, -1))
        
        return importance
    
    def select_blocks(
        self,
        importance: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """
        Select blocks based on importance scores and threshold.
        
        Args:
            importance: Block importance scores [batch, num_blocks, num_blocks]
            threshold: Selection threshold
            
        Returns:
            Binary mask indicating selected blocks [batch, num_blocks, num_blocks]
        """
        # Flatten importance for efficient selection
        batch, num_blocks, _ = importance.shape
        
        # Sort importance values
        importance_flat = importance.view(batch, -1)
        sorted_importance, indices = torch.sort(importance_flat, descending=True)
        
        # Compute cumulative sum
        cumulative_sum = torch.cumsum(sorted_importance, dim=-1)
        total_sum = cumulative_sum[:, -1:]
        
        # Find threshold
        target_sum = total_sum * threshold
        mask_flat = torch.zeros_like(importance_flat)
        
        for b in range(batch):
            # Find how many blocks to select
            selected = torch.where(cumulative_sum[b] <= target_sum[b])[0]
            if len(selected) > 0:
                num_selected = selected[-1] + 1
                mask_flat[b, indices[b, :num_selected]] = 1.0
        
        # Reshape back to block matrix
        mask = mask_flat.view(batch, num_blocks, num_blocks)
        
        return mask
    
    def expand_mask_to_full(
        self,
        block_mask: torch.Tensor,
        seq_len: int,
        block_size: int,
    ) -> torch.Tensor:
        """
        Expand block-level mask to full attention mask.
        
        Args:
            block_mask: Block-level mask [batch, num_blocks, num_blocks]
            seq_len: Full sequence length
            block_size: Size of each block
            
        Returns:
            Full attention mask [batch, seq_len, seq_len]
        """
        batch, num_blocks, _ = block_mask.shape
        full_mask = torch.zeros(batch, seq_len, seq_len, device=block_mask.device)
        
        for b_i in range(num_blocks):
            for b_j in range(num_blocks):
                if block_mask[:, b_i, b_j].any():
                    start_i = b_i * block_size
                    end_i = min((b_i + 1) * block_size, seq_len)
                    start_j = b_j * block_size
                    end_j = min((b_j + 1) * block_size, seq_len)
                    
                    full_mask[:, start_i:end_i, start_j:end_j] = 1.0
        
        return full_mask
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for XAttention.
        
        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]
            v: Value tensor [batch, num_heads, seq_len, head_dim]
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (output, info_dict)
        """
        batch, num_heads, seq_len, head_dim = q.shape
        
        # Reshape for processing
        q_flat = q.view(batch * num_heads, seq_len, head_dim)
        k_flat = k.view(batch * num_heads, seq_len, head_dim)
        v_flat = v.view(batch * num_heads, seq_len, head_dim)
        
        # Compute block importance for each head
        all_masks = []
        densities = []
        
        for h in range(num_heads):
            q_h = q_flat[h::num_heads]
            k_h = k_flat[h::num_heads]
            
            # Compute importance
            importance = self.compute_block_importance(
                q_h, k_h, self.block_size, self.stride
            )
            
            # Get threshold for this head
            if self.use_dynamic_threshold:
                threshold = torch.sigmoid(self.threshold_predictor[h])
            else:
                threshold = self.threshold
            
            # Select blocks
            block_mask = self.select_blocks(importance, threshold)
            
            # Expand to full mask
            full_mask = self.expand_mask_to_full(block_mask, seq_len, self.block_size)
            
            # Apply additional attention mask if provided
            if attention_mask is not None:
                full_mask = full_mask * attention_mask
            
            all_masks.append(full_mask)
            densities.append(block_mask.sum() / (block_mask.shape[1] * block_mask.shape[2]))
        
        # Stack masks
        sparse_mask = torch.stack(all_masks, dim=1)  # [batch, num_heads, seq_len, seq_len]
        
        # Compute sparse attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
        
        # Apply sparse mask
        scores = scores.masked_fill(sparse_mask == 0, float('-inf'))
        
        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # Compute statistics
        density = torch.mean(torch.stack(densities)).item()
        
        info = {
            'density': density,
            'sparse_mask': sparse_mask,
            'threshold': threshold.item() if isinstance(threshold, torch.Tensor) else threshold,
        }
        
        return output, info
    
    def load_weights(self, checkpoint_path: str):
        """Load pre-trained weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        
    def save_weights(self, checkpoint_path: str):
        """Save model weights to checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'hidden_size': self.hidden_size,
                'num_heads': self.num_heads,
                'block_size': self.block_size,
                'stride': self.stride,
                'threshold': self.threshold,
            }
        }, checkpoint_path)