"""
Baseline Full Attention Implementation
This serves as the initialization method before applying XAttention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class BaselineAttention(nn.Module):
    """
    Standard multi-head attention implementation serving as baseline.
    Uses FlashAttention-style implementation for efficiency.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
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
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, L, D)
            mask: Optional attention mask
        Returns:
            Output tensor of shape (B, L, D)
        """
        B, L, D = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, d_h)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out
    
    def get_attention_pattern(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the full attention matrix for analysis
        """
        B, L, D = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        
        return attn  # Shape: (B, H, L, L)
    
    def load_weights(self, state_dict: dict):
        """Load weights from state dictionary"""
        self.load_state_dict(state_dict)
        
    def save_weights(self) -> dict:
        """Save weights to state dictionary"""
        return self.state_dict()


class BaselineAttentionFlash(nn.Module):
    """
    Optimized baseline using FlashAttention-style implementation
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
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
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Optimized forward pass using efficient attention computation
        """
        B, L, D = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Use efficient attention computation
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized attention
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=self.causal
            )
        else:
            # Fallback to manual implementation
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if self.causal:
                causal_mask = torch.triu(
                    torch.ones(L, L, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                scores = scores.masked_fill(causal_mask, float('-inf'))
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            
            attn = F.softmax(scores, dim=-1)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out