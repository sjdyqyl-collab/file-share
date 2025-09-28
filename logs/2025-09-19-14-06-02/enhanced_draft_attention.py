"""
Enhanced DraftAttention with proposed improvements:
1. Dynamic Sparsity Scheduling
2. Multi-Scale Draft Attention
3. Learned Adaptive Pooling
4. Temporal Consistency Module
5. Quantized Sparse Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math
import numpy as np
from draft_attention import DraftAttention


class DynamicSparsityScheduler:
    """
    Dynamic sparsity scheduling that adjusts sparsity ratio during denoising.
    """
    
    def __init__(
        self,
        min_sparsity: float = 0.5,
        max_sparsity: float = 0.9,
        total_steps: int = 50,
        schedule_type: str = "linear"
    ):
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        
    def get_sparsity_ratio(self, step: int) -> float:
        """Get sparsity ratio for current step."""
        if step >= self.total_steps:
            return self.max_sparsity
            
        progress = step / self.total_steps
        
        if self.schedule_type == "linear":
            return self.min_sparsity + (self.max_sparsity - self.min_sparsity) * progress
        elif self.schedule_type == "cosine":
            return self.min_sparsity + (self.max_sparsity - self.min_sparsity) * (1 - math.cos(math.pi * progress)) / 2
        elif self.schedule_type == "exponential":
            return self.min_sparsity * (self.max_sparsity / self.min_sparsity) ** progress
        else:
            return self.min_sparsity


class LearnedAdaptivePooling(nn.Module):
    """
    Learned pooling kernels for adaptive downsampling.
    """
    
    def __init__(
        self,
        in_channels: int,
        kernel_sizes: List[Tuple[int, int]] = [(8, 16), (4, 8), (16, 32)],
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)
        
        # Learnable pooling weights for each scale
        self.pool_weights = nn.ParameterList([
            nn.Parameter(torch.ones(in_channels, 1, kh, kw))
            for kh, kw in kernel_sizes
        ])
        
        # Scale selection weights
        self.scale_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, self.num_scales),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        frame_size: Tuple[int, int], 
        num_frames: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply learned adaptive pooling.
        
        Returns:
            pooled: Downsampled tensor
            scale_weights: Weights for each scale
        """
        B, n, d = x.shape
        H, W = frame_size
        
        # Reshape to spatial-temporal format
        x = x.view(B, num_frames, H, W, d)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(B * num_frames, d, H, W)
        
        # Select scale weights
        scale_weights = self.scale_selector(x.mean(dim=[2, 3]))  # [B*F, num_scales]
        
        # Apply pooling for each scale
        pooled_list = []
        for i, (kernel_weight, (kh, kw)) in enumerate(zip(self.pool_weights, self.kernel_sizes)):
            pooled = F.conv2d(
                x,
                kernel_weight,
                stride=(kh, kw),
                groups=d
            )
            pooled_list.append(pooled)
        
        # Weighted combination
        weighted_pooled = 0
        for i, pooled in enumerate(pooled_list):
            weight = scale_weights[:, i:i+1, None, None]
            weighted_pooled += weight * pooled
        
        # Reshape back to sequence format
        _, _, pooled_H, pooled_W = weighted_pooled.shape
        weighted_pooled = weighted_pooled.view(B, num_frames, d, -1)
        weighted_pooled = weighted_pooled.permute(0, 1, 3, 2).contiguous()
        pooled = weighted_pooled.view(B, -1, d)
        
        return pooled, scale_weights


class MultiScaleDraftAttention(nn.Module):
    """
    Multi-scale draft attention combining multiple pooling scales.
    """
    
    def __init__(
        self,
        scales: List[int] = [64, 128, 256],
        sparsity_ratio: float = 0.9,
    ):
        super().__init__()
        self.scales = scales
        self.sparsity_ratio = sparsity_ratio
        
        # Scale weights for combining attention maps
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))
        
    def _downsample_at_scale(
        self, 
        x: torch.Tensor, 
        scale: int, 
        frame_size: Tuple[int, int], 
        num_frames: int
    ) -> torch.Tensor:
        """Downsample at specific scale factor."""
        B, n, d = x.shape
        H, W = frame_size
        
        # Calculate kernel size based on scale
        kernel_h = max(1, H // (H // math.sqrt(scale)))
        kernel_w = max(1, W // (W // math.sqrt(scale)))
        
        # Reshape and pool
        x = x.view(B, num_frames, H, W, d)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(B * num_frames, d, H, W)
        
        pooled = F.adaptive_avg_pool2d(x, (H // kernel_h, W // kernel_w))
        
        pooled = pooled.view(B, num_frames, d, -1)
        pooled = pooled.permute(0, 1, 3, 2).contiguous()
        pooled = pooled.view(B, -1, d)
        
        return pooled
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        frame_size: Tuple[int, int],
        num_frames: int,
    ) -> torch.Tensor:
        """Compute multi-scale draft attention."""
        attention_maps = []
        
        for scale in self.scales:
            q_scale = self._downsample_at_scale(q, scale, frame_size, num_frames)
            k_scale = self._downsample_at_scale(k, scale, frame_size, num_frames)
            
            d = q_scale.shape[-1]
            scores = torch.bmm(q_scale, k_scale.transpose(-2, -1)) / math.sqrt(d)
            attention = F.softmax(scores, dim=-1)
            attention_maps.append(attention)
        
        # Weighted combination of attention maps
        weights = F.softmax(self.scale_weights, dim=0)
        combined_attention = sum(w * attn for w, attn in zip(weights, attention_maps))
        
        return combined_attention


class TemporalConsistencyModule(nn.Module):
    """
    Temporal consistency module for maintaining coherence across frames.
    """
    
    def __init__(self, d_model: int, num_frames: int):
        super().__init__()
        self.d_model = d_model
        self.num_frames = num_frames
        
        # Temporal attention layers
        self.temporal_q = nn.Linear(d_model, d_model)
        self.temporal_k = nn.Linear(d_model, d_model)
        self.temporal_v = nn.Linear(d_model, d_model)
        
        # Spatial-temporal fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        frame_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Apply temporal consistency.
        
        Args:
            x: [B, n, d] where n = F*H*W
            
        Returns:
            Temporally consistent features [B, n, d]
        """
        B, n, d = x.shape
        H, W = frame_size
        
        # Reshape to separate spatial and temporal dimensions
        x_spatial = x.view(B, self.num_frames, H * W, d)
        
        # Global average pooling per frame
        x_global = x_spatial.mean(dim=2)  # [B, F, d]
        
        # Temporal attention
        q = self.temporal_q(x_global)  # [B, F, d]
        k = self.temporal_k(x_global)  # [B, F, d]
        v = self.temporal_v(x_global)  # [B, F, d]
        
        scores = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(d)
        temporal_attn = F.softmax(scores, dim=-1)
        
        # Apply temporal attention
        x_temporal = torch.bmm(temporal_attn, v)  # [B, F, d]
        
        # Broadcast back to spatial dimensions
        x_temporal = x_temporal.unsqueeze(2).expand(-1, -1, H * W, -1)
        x_temporal = x_temporal.reshape(B, n, d)
        
        # Fuse spatial and temporal information
        x_fused = torch.cat([x, x_temporal], dim=-1)
        x_out = self.fusion(x_fused)
        
        return x_out


class QuantizedSparseAttention(nn.Module):
    """
    Quantized sparse attention for memory efficiency.
    """
    
    def __init__(self, d_model: int, num_bits: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_bits = num_bits
        
        # Quantization parameters
        self.register_buffer('q_scale', torch.tensor(1.0))
        self.register_buffer('q_zero_point', torch.tensor(0.0))
        
    def quantize_tensor(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Quantize tensor to int8."""
        x_min = x.min()
        x_max = x.max()
        
        qmin = -(2 ** (self.num_bits - 1))
        qmax = 2 ** (self.num_bits - 1) - 1
        
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = qmin - torch.round(x_min / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax)
        
        q_x = torch.round(x / scale + zero_point)
        q_x = torch.clamp(q_x, qmin, qmax)
        
        return q_x.to(torch.int8), scale.item(), zero_point.item()
    
    def dequantize_tensor(self, q_x: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
        """Dequantize int8 tensor back to float."""
        return (q_x.float() - zero_point) * scale
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sparsity_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply quantized sparse attention."""
        # Quantize inputs
        q_q, q_scale, q_zp = self.quantize_tensor(q)
        k_q, k_scale, k_zp = self.quantize_tensor(k)
        v_q, v_scale, v_zp = self.quantize_tensor(v)
        
        # Dequantize for computation (simulating int8 matmul)
        q_float = self.dequantize_tensor(q_q, q_scale, q_zp)
        k_float = self.dequantize_tensor(k_q, k_scale, k_zp)
        v_float = self.dequantize_tensor(v_q, v_scale, v_zp)
        
        # Compute attention with sparsity
        d = q.shape[-1]
        scores = torch.bmm(q_float, k_float.transpose(-2, -1)) / math.sqrt(d)
        scores = scores.masked_fill(sparsity_mask == 0, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        attention = attention * sparsity_mask
        
        return torch.bmm(attention, v_float)


class EnhancedDraftAttention(nn.Module):
    """
    Enhanced DraftAttention with all proposed improvements.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        sparsity_ratio: float = 0.9,
        kernel_size: Tuple[int, int] = (8, 16),
        use_dynamic_sparsity: bool = True,
        use_multi_scale: bool = True,
        use_learned_pooling: bool = True,
        use_temporal_consistency: bool = True,
        use_quantization: bool = False,
        num_frames: int = 16,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_dynamic_sparsity = use_dynamic_sparsity
        self.use_multi_scale = use_multi_scale
        self.use_learned_pooling = use_learned_pooling
        self.use_temporal_consistency = use_temporal_consistency
        self.use_quantization = use_quantization
        
        # Core components
        self.draft_attention = DraftAttention(
            sparsity_ratio=sparsity_ratio,
            kernel_size=kernel_size
        )
        
        # Enhanced components
        if use_dynamic_sparsity:
            self.sparsity_scheduler = DynamicSparsityScheduler()
            
        if use_multi_scale:
            self.multi_scale_attention = MultiScaleDraftAttention()
            
        if use_learned_pooling:
            self.learned_pooling = LearnedAdaptivePooling(d_model)
            
        if use_temporal_consistency:
            self.temporal_consistency = TemporalConsistencyModule(d_model, num_frames)
            
        if use_quantization:
            self.quantized_attention = QuantizedSparseAttention(d_model)
            
        # Learnable fusion weights
        if use_multi_scale:
            self.fusion_weights = nn.Parameter(torch.ones(3))  # draft, multi-scale, temporal
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        frame_size: Tuple[int, int],
        num_frames: int,
        step: int = 0,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of EnhancedDraftAttention.
        """
        B, n, d = q.shape
        
        # Dynamic sparsity scheduling
        if self.use_dynamic_sparsity:
            current_sparsity = self.sparsity_scheduler.get_sparsity_ratio(step)
            self.draft_attention.sparsity_ratio = current_sparsity
        
        # Multi-scale draft attention
        if self.use_multi_scale:
            multi_scale_attention = self.multi_scale_attention(
                q, k, v, frame_size, num_frames
            )
        else:
            multi_scale_attention = None
        
        # Learned adaptive pooling
        if self.use_learned_pooling:
            q_pooled, _ = self.learned_pooling(q, frame_size, num_frames)
            k_pooled, _ = self.learned_pooling(k, frame_size, num_frames)
        else:
            q_pooled = self.draft_attention._downsample_with_pooling(q, frame_size, num_frames)
            k_pooled = self.draft_attention._downsample_with_pooling(k, frame_size, num_frames)
        
        # Compute draft attention
        draft_attention = self.draft_attention._compute_draft_attention(q_pooled, k_pooled, k_pooled)
        
        # Generate sparsity mask
        sparsity_mask = self.draft_attention._generate_sparsity_mask(
            draft_attention, 
            self.draft_attention.sparsity_ratio
        )
        
        # Extend to full resolution
        full_sparsity_mask = self.draft_attention._extend_mask_to_full_resolution(
            sparsity_mask,
            n,
            frame_size,
            num_frames
        )
        
        # Apply sparsity with optional quantization
        if self.use_quantization:
            out = self.quantized_attention(q, k, v, full_sparsity_mask)
        else:
            scores = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(d)
            scores = scores.masked_fill(full_sparsity_mask == 0, float('-inf'))
            attention = F.softmax(scores, dim=-1)
            attention = attention * full_sparsity_mask
            out = torch.bmm(attention, v)
        
        # Temporal consistency
        if self.use_temporal_consistency:
            out = self.temporal_consistency(out, frame_size)
        
        return out
    
    def load_weights(self, state_dict: dict):
        """Load weights for enhanced components."""
        self.load_state_dict(state_dict, strict=False)
    
    def save_weights(self) -> dict:
        """Save weights for enhanced components."""
        return self.state_dict()


class EnhancedDraftAttentionBlock(nn.Module):
    """
    Multi-head attention block using EnhancedDraftAttention.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        sparsity_ratio: float = 0.9,
        kernel_size: Tuple[int, int] = (8, 16),
        use_dynamic_sparsity: bool = True,
        use_multi_scale: bool = True,
        use_learned_pooling: bool = True,
        use_temporal_consistency: bool = True,
        use_quantization: bool = False,
        num_frames: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.enhanced_attention = EnhancedDraftAttention(
            d_model=d_model,
            n_heads=n_heads,
            sparsity_ratio=sparsity_ratio,
            kernel_size=kernel_size,
            use_dynamic_sparsity=use_dynamic_sparsity,
            use_multi_scale=use_multi_scale,
            use_learned_pooling=use_learned_pooling,
            use_temporal_consistency=use_temporal_consistency,
            use_quantization=use_quantization,
            num_frames=num_frames,
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        frame_size: Tuple[int, int],
        num_frames: int,
        step: int = 0,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, n, d_model]
            frame_size: (H, W) spatial dimensions
            num_frames: Number of temporal frames
            step: Current denoising step
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [B, n, d_model]
        """
        B, n, d = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply enhanced attention per head
        out = []
        for head_idx in range(self.n_heads):
            head_q = q[:, head_idx]  # [B, n, d_head]
            head_k = k[:, head_idx]  # [B, n, d_head]
            head_v = v[:, head_idx]  # [B, n, d_head]
            
            head_out = self.enhanced_attention(
                head_q, head_k, head_v,
                frame_size, num_frames, step, attention_mask
            )
            out.append(head_out)
        
        # Concatenate heads
        out = torch.stack(out, dim=1)  # [B, n_heads, n, d_head]
        out = out.transpose(1, 2).contiguous().view(B, n, d)
        
        # Final projection
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out