"""
EnhancedDraftAttention: Improved version of DraftAttention with suggested enhancements
including multi-scale adaptive pooling, quantization, dynamic sparsity, and motion-aware features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math
import numpy as np


class QuantizedLinear(nn.Module):
    """Linear layer with INT4/INT8 quantization support."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False, bits: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Full precision weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        # Quantization parameters
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('weight_zero_point', torch.zeros(1))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.weight)
    
    def quantize_weights(self):
        """Quantize weights to specified bit width."""
        if self.bits == 8:
            qmin, qmax = -128, 127
        elif self.bits == 4:
            qmin, qmax = -8, 7
        else:
            raise ValueError(f"Unsupported bit width: {self.bits}")
            
        # Calculate scale and zero point
        w_min = self.weight.min()
        w_max = self.weight.max()
        
        self.weight_scale = (w_max - w_min) / (qmax - qmin)
        self.weight_zero_point = qmin - (w_min / self.weight_scale).round()
        
        # Quantize
        quantized_weight = torch.round(self.weight / self.weight_scale + self.weight_zero_point)
        quantized_weight = torch.clamp(quantized_weight, qmin, qmax)
        
        return quantized_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization."""
        if self.training:
            # Use full precision during training
            weight = self.weight
        else:
            # Use quantized weights during inference
            quantized_weight = self.quantize_weights()
            weight = quantized_weight * self.weight_scale - self.weight_zero_point * self.weight_scale
            
        return F.linear(x, weight, self.bias)


class MultiScaleDraftAttention(nn.Module):
    """Multi-scale draft attention with adaptive kernel selection."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        scales: List[Tuple[int, int]] = [(4, 8), (8, 16), (16, 32)],
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scales = scales
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Scale selection network
        self.scale_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, len(scales)),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        height: int, 
        width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-scale draft attention and select best scale.
        
        Args:
            q: Query tensor [B, H, N, D]
            k: Key tensor [B, H, N, D]
            height: Height of spatial feature map
            width: Width of spatial feature map
            
        Returns:
            Selected draft attention map and corresponding kernel
        """
        B, H, N, D = q.shape
        
        # Global feature for scale selection
        global_feat = q.mean(dim=[1, 2])  # [B, D]
        scale_weights = self.scale_selector(global_feat.unsqueeze(-1).unsqueeze(-1))
        
        # Compute draft attention for each scale
        draft_attentions = []
        for kernel in self.scales:
            draft_attn = self._compute_single_scale_draft(q, k, height, width, kernel)
            draft_attentions.append(draft_attn)
        
        # Weighted combination of scales
        combined_draft = torch.zeros_like(draft_attentions[0])
        for i, (draft, weight) in enumerate(zip(draft_attentions, scale_weights.unbind(-1))):
            combined_draft += draft * weight.view(B, 1, 1, 1)
        
        # Select dominant scale
        dominant_scale_idx = scale_weights.argmax(dim=-1)
        selected_kernel = self.scales[dominant_scale_idx.item()]
        
        return combined_draft, selected_kernel
    
    def _compute_single_scale_draft(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        height: int, 
        width: int,
        kernel: Tuple[int, int]
    ) -> torch.Tensor:
        """Compute draft attention for a single scale."""
        B, H, N, D = q.shape
        
        q_spatial = q.view(B, H, height, width, D)
        k_spatial = k.view(B, H, height, width, D)
        
        q_draft = F.avg_pool2d(
            q_spatial.permute(0, 1, 4, 2, 3).reshape(B * H, D, height, width),
            kernel_size=kernel,
            stride=kernel
        ).view(B, H, D, -1).permute(0, 1, 3, 2)
        
        k_draft = F.avg_pool2d(
            k_spatial.permute(0, 1, 4, 2, 3).reshape(B * H, D, height, width),
            kernel_size=kernel,
            stride=kernel
        ).view(B, H, D, -1).permute(0, 1, 3, 2)
        
        draft_attention = torch.matmul(q_draft, k_draft.transpose(-2, -1)) / math.sqrt(D)
        draft_attention = F.softmax(draft_attention, dim=-1)
        
        return draft_attention


class MotionAwareSparsity(nn.Module):
    """Motion-aware sparsity that identifies static regions."""
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Simple optical flow estimation network
        self.flow_estimator = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),  # 6 channels for two frames
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1)  # 2 channels for flow
        )
    
    def estimate_motion_mask(
        self, 
        x: torch.Tensor, 
        height: int, 
        width: int,
        threshold: float = 0.1
    ) -> torch.Tensor:
        """
        Estimate motion mask from temporal features.
        
        Args:
            x: Input tensor [B, N, D] with temporal dimension
            height: Height of spatial feature map
            width: Width of spatial feature map
            threshold: Motion detection threshold
            
        Returns:
            Motion mask [B, N] where 1 indicates motion, 0 static
        """
        B, N, D = x.shape
        
        # Assume temporal dimension is along sequence
        frames = N // (height * width)
        if frames < 2:
            return torch.ones(B, N, device=self.device)
        
        # Reshape to [B, T, H, W, D]
        x_reshaped = x.view(B, frames, height, width, D)
        
        # Compute motion between consecutive frames
        motion_masks = []
        for t in range(frames - 1):
            frame1 = x_reshaped[:, t].permute(0, 3, 1, 2)  # [B, D, H, W]
            frame2 = x_reshaped[:, t+1].permute(0, 3, 1, 2)
            
            # Simple flow estimation
            flow_input = torch.cat([frame1, frame2], dim=1)
            flow = self.flow_estimator(flow_input)
            
            # Motion magnitude
            motion_magnitude = torch.norm(flow, dim=1)  # [B, H, W]
            
            # Threshold to get binary mask
            motion_mask = (motion_magnitude > threshold).float()
            motion_masks.append(motion_mask.view(B, -1))
        
        # Use maximum motion across frames
        motion_mask = torch.stack(motion_masks, dim=1).max(dim=1)[0]
        
        return motion_mask


class EnhancedDraftAttention(nn.Module):
    """
    Enhanced DraftAttention with multi-scale, quantization, dynamic sparsity,
    and motion-aware features.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        base_sparsity_ratio: float = 0.75,
        scales: List[Tuple[int, int]] = [(4, 8), (8, 16), (16, 32)],
        quantization_bits: int = 8,
        use_motion_aware: bool = True,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.base_sparsity_ratio = base_sparsity_ratio
        self.quantization_bits = quantization_bits
        self.use_motion_aware = use_motion_aware
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Quantized linear projections
        self.q_proj = QuantizedLinear(dim, dim, bias=False, bits=quantization_bits)
        self.k_proj = QuantizedLinear(dim, dim, bias=False, bits=quantization_bits)
        self.v_proj = QuantizedLinear(dim, dim, bias=False, bits=quantization_bits)
        self.out_proj = QuantizedLinear(dim, dim, bias=False, bits=quantization_bits)
        
        # Multi-scale draft attention
        self.multi_scale_draft = MultiScaleDraftAttention(dim, num_heads, scales, device)
        
        # Motion-aware sparsity
        if use_motion_aware:
            self.motion_sparsity = MotionAwareSparsity(device)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            if hasattr(module, '_init_weights'):
                module._init_weights()
    
    def _dynamic_sparsity_schedule(self, step_ratio: float) -> float:
        """
        Dynamic sparsity scheduling.
        
        Args:
            step_ratio: Current denoising step ratio (0-1)
            
        Returns:
            Adjusted sparsity ratio
        """
        # Start with high sparsity, gradually reduce
        if step_ratio < 0.3:
            return min(0.9, self.base_sparsity_ratio + 0.15)
        elif step_ratio < 0.7:
            return self.base_sparsity_ratio
        else:
            return max(0.5, self.base_sparsity_ratio - 0.25)
    
    def _compute_enhanced_sparsity_mask(
        self,
        draft_attention: torch.Tensor,
        height: int,
        width: int,
        x: torch.Tensor,
        step_ratio: float
    ) -> torch.Tensor:
        """
        Compute enhanced sparsity mask with motion awareness.
        
        Args:
            draft_attention: Draft attention map
            height: Height of spatial feature map
            width: Width of spatial feature map
            x: Input tensor for motion estimation
            step_ratio: Current denoising step ratio
            
        Returns:
            Enhanced sparsity mask
        """
        B, H, g, _ = draft_attention.shape
        
        # Get dynamic sparsity ratio
        current_sparsity = self._dynamic_sparsity_schedule(step_ratio)
        
        # Base sparsity mask from draft attention
        draft_mean = draft_attention.mean(dim=1)
        num_keep = max(1, int(g * g * current_sparsity))
        
        flat_attention = draft_mean.view(B, -1)
        _, top_indices = torch.topk(flat_attention, num_keep, dim=-1)
        
        region_mask = torch.zeros_like(flat_attention)
        region_mask.scatter_(1, top_indices, 1.0)
        region_mask = region_mask.view(B, g, g)
        
        # Apply motion-aware adjustments
        if self.use_motion_aware and x is not None:
            motion_mask = self.motion_sparsity.estimate_motion_mask(x, height, width)
            
            # Downsample motion mask to region level
            motion_regions = motion_mask.view(B, g, -1).mean(dim=-1)  # [B, g]
            motion_regions = motion_regions.unsqueeze(-1) @ motion_regions.unsqueeze(-2)  # [B, g, g]
            
            # Increase sparsity in static regions
            static_threshold = 0.1
            static_regions = (motion_regions < static_threshold).float()
            
            # Reduce sparsity in static regions (keep more tokens)
            region_mask = region_mask * (1 - 0.3 * static_regions)
        
        # Expand to full resolution
        h_patches = height // 8  # Use base 8x16 kernel
        w_patches = width // 16
        
        full_mask = region_mask.repeat_interleave(8, dim=1)
        full_mask = full_mask.repeat_interleave(16, dim=2)
        
        full_mask = full_mask.view(B, -1, 1)
        full_mask = full_mask @ full_mask.transpose(-2, -1)
        
        return full_mask.unsqueeze(1)
    
    def forward(
        self,
        x: torch.Tensor,
        height: int,
        width: int,
        step_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Forward pass of EnhancedDraftAttention.
        
        Args:
            x: Input tensor [B, N, D]
            height: Height of spatial feature map
            width: Width of spatial feature map
            step_ratio: Current denoising step ratio (0-1)
            
        Returns:
            Output tensor [B, N, D]
        """
        B, N, D = x.shape
        
        # Compute Q, K, V with quantization
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Multi-scale draft attention
        draft_attention, selected_kernel = self.multi_scale_draft(q, k, height, width)
        
        # Enhanced sparsity mask
        sparsity_mask = self._compute_enhanced_sparsity_mask(
            draft_attention, height, width, x, step_ratio
        )
        
        # Efficient attention computation with quantization
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply sparsity mask
        mask = sparsity_mask.expand(-1, self.num_heads, -1, -1)
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights.masked_fill(mask == 0, 0.0)
        
        # Apply to values
        out = torch.matmul(attention_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)
    
    def load_weights(self, state_dict: dict):
        """Load pre-trained weights."""
        self.load_state_dict(state_dict)
    
    def save_weights(self, path: str):
        """Save model weights."""
        torch.save(self.state_dict(), path)
    
    def set_quantization_bits(self, bits: int):
        """Update quantization bit width."""
        self.quantization_bits = bits
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            module.bits = bits
    
    def set_sparsity_ratio(self, ratio: float):
        """Update base sparsity ratio."""
        self.base_sparsity_ratio = ratio