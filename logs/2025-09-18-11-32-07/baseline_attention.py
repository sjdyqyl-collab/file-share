"""
Baseline Full Attention Implementation
This serves as the reference implementation for comparison with XAttention methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BaselineAttention(nn.Module):
    """
    Standard multi-head attention implementation for comparison purposes.
    This represents the full attention baseline that XAttention aims to accelerate.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        """
        Initialize baseline attention.
        
        Args:
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of baseline attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)
        
        return output
    
    def get_attention_matrix(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get the full attention matrix for analysis purposes.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            mask: Optional attention mask
            
        Returns:
            Attention matrix of shape [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        return attn_weights
    
    def get_flops(self, seq_len: int) -> int:
        """
        Calculate FLOPs for full attention computation.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Number of FLOPs
        """
        # QK^T: [L, d] @ [d, L] = [L, L] -> L*d*L multiplications + L*d*L additions
        # Attention @ V: [L, L] @ [L, d] = [L, d] -> L*L*d multiplications + L*L*d additions
        # Total: 2*L*L*d + 2*L*L*d = 4*L*L*d
        return 4 * seq_len * seq_len * self.hidden_size


class BaselineAttentionWithCache(nn.Module):
    """
    Baseline attention with KV caching for autoregressive generation.
    This serves as reference for the decoding stage extension in improved XAttention.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        
        # Cache for KV pairs
        self.k_cache = None
        self.v_cache = None
        self.cache_length = 0
        
    def forward(self, x: torch.Tensor, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV caching.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            use_cache: Whether to use and update KV cache
            
        Returns:
            Output tensor and optional updated cache
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if use_cache and self.k_cache is not None:
            # Concatenate with cached values
            k = torch.cat([self.k_cache, k], dim=2)
            v = torch.cat([self.v_cache, v], dim=2)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask for autoregressive generation
        if use_cache:
            causal_mask = torch.triu(torch.ones(seq_len, k.size(2)), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)
        
        if use_cache:
            # Update cache
            self.k_cache = k
            self.v_cache = v
            self.cache_length = k.size(2)
            return output, (self.k_cache, self.v_cache)
        else:
            return output, None
    
    def clear_cache(self):
        """Clear the KV cache."""
        self.k_cache = None
        self.v_cache = None
        self.cache_length = 0


def test_baseline_attention():
    """Test the baseline attention implementation."""
    torch.manual_seed(42)
    
    batch_size, seq_len, hidden_size, num_heads = 2, 128, 256, 8
    
    # Create attention module
    attention = BaselineAttention(hidden_size, num_heads)
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass
    output = attention(x)
    
    # Get attention matrix
    attn_matrix = attention.get_attention_matrix(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention matrix shape: {attn_matrix.shape}")
    print(f"FLOPs for sequence length {seq_len}: {attention.get_flops(seq_len):,}")
    
    # Test with cache
    attention_cache = BaselineAttentionWithCache(hidden_size, num_heads)
    
    # Simulate autoregressive generation
    output1, cache1 = attention_cache(x[:, :64, :], use_cache=True)
    output2, cache2 = attention_cache(x[:, 64:, :], use_cache=True)
    
    print(f"Cached generation - Step 1 output shape: {output1.shape}")
    print(f"Cached generation - Step 2 output shape: {output2.shape}")
    print(f"Cache length: {attention_cache.cache_length}")
    
    attention_cache.clear_cache()
    print("Cache cleared successfully!")


if __name__ == "__main__":
    test_baseline_attention()