import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class BaseAttention(nn.Module):
    """
    Standard multi-head attention implementation for comparison.
    
    Shape conventions:
    - B: batch size
    - L: sequence length
    - H: hidden size (d_model)
    - num_heads: number of attention heads
    - head_dim: dimension per head (H // num_heads)
    """
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False
    ) -> torch.Tensor:
        """
        Standard multi-head attention forward pass.
        
        Args:
            query: [B, L, H]
            key: [B, L, H]
            value: [B, L, H]
            attention_mask: [B, L, L] or None
            causal: whether to apply causal masking
            
        Returns:
            output: [B, L, H]
        """
        B, L, H = query.shape
        
        # Linear projections
        Q = self.q_proj(query)  # [B, L, H]
        K = self.k_proj(key)    # [B, L, H]
        V = self.v_proj(value)  # [B, L, H]
        
        # Reshape for multi-head attention
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, num_heads, L, L]
        
        # Apply causal mask if needed
        if causal:
            causal_mask = torch.triu(torch.ones(L, L, device=query.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L, L]
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # [B, num_heads, L, head_dim]
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, L, H)  # [B, L, H]
        out = self.out_proj(out)
        
        return out, attn_weights
    
    def load_weights(self, state_dict: dict):
        """Load pre-trained weights if available."""
        self.load_state_dict(state_dict)