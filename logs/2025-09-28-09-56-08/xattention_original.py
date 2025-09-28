import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class XAttentionOriginal(nn.Module):
    """
    Original XAttention implementation with antidiagonal scoring and dynamic threshold prediction.
    
    Paper: XAttention: Block Sparse Attention with Antidiagonal Scoring
    """
    
    def __init__(self, 
                 hidden_size: int,
                 num_heads: int,
                 block_size: int = 8,
                 stride: int = 8,
                 head_dim: Optional[int] = None,
                 dropout: float = 0.0):
        """
        Initialize XAttention.
        
        Args:
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
            block_size: Size of attention blocks (B×B)
            stride: Stride parameter for antidiagonal scoring
            head_dim: Dimension per head (defaults to hidden_size // num_heads)
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.block_size = block_size
        self.stride = stride
        self.head_dim = head_dim or hidden_size // num_heads
        self.dropout = dropout
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Initialize thresholds (will be optimized dynamically)
        self.register_buffer('thresholds', torch.ones(num_heads) * 0.9)
        
    def reshape_for_antidiagonal(self, x: torch.Tensor, stride: int) -> torch.Tensor:
        """
        Reshape tensor to extract antidiagonal elements with stride.
        
        Args:
            x: Input tensor of shape [B, L, D] or [L, D]
            stride: Stride parameter
            
        Returns:
            Reshaped tensor for antidiagonal computation
        """
        if x.dim() == 3:
            B, L, D = x.shape
            # For batched input, process each batch separately
            reshaped = []
            for b in range(B):
                x_slice = x[b]  # [L, D]
                reshaped_batch = []
                # Extract antidiagonal elements with stride
                for i in range(stride-1, -1, -1):
                    reshaped_batch.append(x_slice[i::stride, :])
                reshaped.append(torch.cat(reshaped_batch, dim=0))
            return torch.stack(reshaped, dim=0)
        else:
            # Single batch case
            L, D = x.shape
            reshaped = []
            for i in range(stride-1, -1, -1):
                reshaped.append(x[i::stride, :])
            return torch.cat(reshaped, dim=0)
    
    def compute_antidiagonal_scores(self, 
                                    Q_block: torch.Tensor, 
                                    K_block: torch.Tensor,
                                    head_idx: int) -> torch.Tensor:
        """
        Compute antidiagonal scores for a block of attention.
        
        Args:
            Q_block: Query block [B, B, head_dim] or [B, head_dim]
            K_block: Key block [L, head_dim] 
            head_idx: Head index for threshold selection
            
        Returns:
            Antidiagonal scores [B, B]
        """
        B = Q_block.size(-2) if Q_block.dim() == 3 else Q_block.size(0)
        
        # Reshape for antidiagonal computation
        Q_reshaped = self.reshape_for_antidiagonal(Q_block, self.stride)
        K_reshaped = self.reshape_for_antidiagonal(K_block, self.stride)
        
        # Compute approximate attention scores
        scale = 1.0 / np.sqrt(self.head_dim * self.stride)
        scores = torch.matmul(Q_reshaped, K_reshaped.transpose(-2, -1)) * scale
        
        # Apply softmax to get probabilities
        attention_probs = F.softmax(scores, dim=-1)
        
        # Map back to original block size by summing antidiagonal probabilities
        if attention_probs.dim() == 3:
            batch_size = attention_probs.size(0)
            block_scores = torch.zeros(batch_size, B, B, device=attention_probs.device)
            
            for b in range(batch_size):
                # Reconstruct full block from antidiagonal pattern
                idx = 0
                for i in range(self.stride):
                    for j in range(i, B, self.stride):
                        if idx < attention_probs.size(1):
                        # Distribute antidiagonal scores back to original positions
                            for k in range(min(self.stride, B - j)):
                                if j + k < B and idx + k < attention_probs.size(2):
                                    block_scores[b, j + k, (j + k + i) % B] = attention_probs[b, idx, idx + k]
                            idx += 1
        else:
            block_scores = torch.zeros(B, B, device=attention_probs.device)
            idx = 0
            for i in range(self.stride):
                for j in range(i, B, self.stride):
                    if idx < attention_probs.size(0):
                        # Distribute antidiagonal scores back to original positions
                        for k in range(min(self.stride, B - j)):
                            if j + k < B and idx + k < attention_probs.size(1):
                                block_scores[j + k, (j + k + i) % B] = attention_probs[idx, idx + k]
                        idx += 1
        
        return block_scores
    
    def select_blocks_threshold(self, scores: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Select blocks using threshold-based approach.
        
        Args:
            scores: Block scores [B, B]
            threshold: Selection threshold
            
        Returns:
            Binary mask [B, B] indicating selected blocks
        """
        # Flatten scores and compute cumulative sum
        flat_scores = scores.flatten()
        sorted_scores, sorted_indices = torch.sort(flat_scores, descending=True)
        
        # Compute cumulative sum
        cumulative_sum = torch.cumsum(sorted_scores, dim=0)
        total_sum = cumulative_sum[-1]
        
        # Find minimal set of blocks that exceed threshold
        target_sum = total_sum * threshold
        mask = torch.zeros_like(flat_scores, dtype=torch.bool)
        
        # Select blocks until threshold is reached
        for i in range(len(sorted_scores)):
            if cumulative_sum[i] >= target_sum:
                selected_indices = sorted_indices[:i+1]
                mask[selected_indices] = True
                break
        
        return mask.reshape(scores.shape)
    
    def dynamic_threshold_prediction(self, 
                                   Q: torch.Tensor, 
                                   K: torch.Tensor,
                                   num_combinations: int = 1000) -> torch.Tensor:
        """
        Optimize thresholds for each head using dynamic programming approach.
        
        Args:
            Q: Queries [B, H, L, head_dim]
            K: Keys [B, H, L, head_dim]
            num_combinations: Number of threshold combinations to explore
            
        Returns:
            Optimized thresholds [H]
        """
        B, H, L, _ = Q.shape
        
        # Initialize DP table
        dp_table = torch.zeros(H, num_combinations, device=Q.device)
        
        # Explore different threshold combinations
        for h in range(H):
            for m in range(num_combinations):
                # Current threshold (decreasing from 0.9)
                current_threshold = 0.9 * (0.9 ** (m / 100))
                
                # Evaluate performance with current threshold
                # (Simplified: use attention entropy as proxy for performance)
                head_q = Q[:, h, :, :]  # [B, L, head_dim]
                head_k = K[:, h, :, :]  # [B, L, head_dim]
                
                # Compute approximate attention for evaluation
                scale = 1.0 / np.sqrt(self.head_dim)
                attn_scores = torch.matmul(head_q, head_k.transpose(-2, -1)) * scale
                attn_probs = F.softmax(attn_scores, dim=-1)
                
                # Use entropy as performance metric (lower entropy = more focused attention)
                entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8), dim=-1).mean()
                
                # Store performance score (negative entropy for maximization)
                if m == 0:
                    dp_table[h, m] = -entropy
                else:
                    dp_table[h, m] = max(dp_table[h, m-1], -entropy)
        
        # Select optimal thresholds
        optimal_thresholds = torch.zeros(H, device=Q.device)
        for h in range(H):
            best_idx = torch.argmax(dp_table[h])
            optimal_thresholds[h] = 0.9 * (0.9 ** (best_idx.item() / 100))
        
        return optimal_thresholds
    
    def sparse_attention(self, 
                        Q: torch.Tensor, 
                        K: torch.Tensor, 
                        V: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        """
        Compute sparse attention using the selected mask.
        
        Args:
            Q: Queries [B, H, L, head_dim]
            K: Keys [B, H, L, head_dim]
            V: Values [B, H, L, head_dim]
            mask: Sparse mask [B, H, L, L]
            
        Returns:
            Attention output [B, H, L, head_dim]
        """
        B, H, L, _ = Q.shape
        
        # Apply mask to attention scores
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, L, L]
        
        # Apply sparse mask
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Compute attention probabilities
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        
        # Apply attention to values
        output = torch.matmul(attn_probs, V)  # [B, H, L, head_dim]
        
        return output
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                optimize_thresholds: bool = False) -> torch.Tensor:
        """
        Forward pass of XAttention.
        
        Args:
            hidden_states: Input tensor [B, L, hidden_size]
            attention_mask: Optional attention mask
            optimize_thresholds: Whether to optimize thresholds during this forward pass
            
        Returns:
            Output tensor [B, L, hidden_size]
        """
        B, L, _ = hidden_states.shape
        
        # Project to Q, K, V
        Q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Optimize thresholds if requested
        if optimize_thresholds:
            self.thresholds = self.dynamic_threshold_prediction(Q, K)
        
        # Initialize sparse mask
        sparse_mask = torch.zeros(B, self.num_heads, L, L, dtype=torch.bool, device=hidden_states.device)
        
        # Process each head separately
        for h in range(self.num_heads):
            head_q = Q[:, h, :, :]  # [B, L, head_dim]
            head_k = K[:, h, :, :]  # [B, L, head_dim]
            threshold = self.thresholds[h]
            
            # Process in blocks
            num_blocks = L // self.block_size
            
            for b in range(num_blocks):
                start_idx = b * self.block_size
                end_idx = min((b + 1) * self.block_size, L)
                
                # Extract blocks
                q_block = head_q[:, start_idx:end_idx, :]  # [B, block_size, head_dim]
                k_block = head_k  # [B, L, head_dim]
                
                # Compute antidiagonal scores
                block_scores = self.compute_antidiagonal_scores(q_block, k_block, h)
                
                # Select blocks using threshold
                if block_scores.dim() == 3:
                    # Batched case - process each batch separately
                    for batch_idx in range(B):
                        batch_scores = block_scores[batch_idx]  # [block_size, block_size]
                        block_mask = self.select_blocks_threshold(batch_scores, threshold)
                        
                        # Update sparse mask
                        sparse_mask[batch_idx, h, start_idx:end_idx, start_idx:end_idx] = block_mask
                else:
                    # Single batch case
                    block_mask = self.select_blocks_threshold(block_scores, threshold)
                    sparse_mask[:, h, start_idx:end_idx, start_idx:end_idx] = block_mask
        
        # Apply attention mask if provided
        if attention_mask is not None:
            sparse_mask = sparse_mask & attention_mask.unsqueeze(1).unsqueeze(1)
        
        # Compute sparse attention
        attn_output = self.sparse_attention(Q, K, V, sparse_mask)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        output = self.out_proj(attn_output)
        
        return output
    
    def load_weights(self, state_dict: dict):
        """Load pre-trained weights."""
        self.load_state_dict(state_dict)
    
    def get_sparsity_stats(self) -> dict:
        """Get sparsity statistics."""
        return {
            'thresholds': self.thresholds.cpu().numpy(),
            'average_threshold': self.thresholds.mean().item(),
            'min_threshold': self.thresholds.min().item(),
            'max_threshold': self.thresholds.max().item()
        }