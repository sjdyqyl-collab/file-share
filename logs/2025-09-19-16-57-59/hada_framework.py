"""
HADA: Hierarchical Adaptive Draft Attention Framework
Advanced implementation incorporating all proposed improvements from gaps_improvements.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math
import numpy as np
from draft_attention import DraftAttention


class MultiScaleDraftAttention(nn.Module):
    """Adaptive Multi-Scale Draft Attention (AMDA) component."""
    
    def __init__(
        self,
        dim: int,
        scales: List[Tuple[int, int]] = [(4, 4), (8, 8), (16, 16), (32, 32)],
        device: str = "cuda"
    ):
        super().__init__()
        self.dim = dim
        self.scales = scales
        self.num_scales = len(scales)
        self.device = device
        
        # Importance weight predictor
        self.importance_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, self.num_scales),
            nn.Softmax(dim=-1)
        )
        
    def _compute_scale_draft(self, 
                           q: torch.Tensor, 
                           k: torch.Tensor,
                           scale: Tuple[int, int],
                           frame_size: Tuple[int, int],
                           num_frames: int) -> torch.Tensor:
        """Compute draft attention for a specific scale."""
        B, N, D = q.shape
        H, W = frame_size
        
        # Reshape for pooling
        q_reshaped = q.view(B, num_frames, H, W, D).permute(0, 4, 1, 2, 3)
        k_reshaped = k.view(B, num_frames, H, W, D).permute(0, 4, 1, 2, 3)
        
        # Average pooling
        q_pooled = F.avg_pool3d(
            q_reshaped,
            kernel_size=(1, *scale),
            stride=(1, *scale)
        )
        k_pooled = F.avg_pool3d(
            k_reshaped,
            kernel_size=(1, *scale),
            stride=(1, *scale)
        )
        
        # Reshape back
        B, D, F, H_pooled, W_pooled = q_pooled.shape
        q_draft = q_pooled.permute(0, 2, 3, 4, 1).reshape(B, -1, D)
        k_draft = k_pooled.permute(0, 2, 3, 4, 1).reshape(B, -1, D)
        
        # Compute attention
        scale_factor = 1.0 / math.sqrt(D)
        attn_scores = torch.bmm(q_draft, k_draft.transpose(-2, -1)) * scale_factor
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        return attn_probs
    
    def forward(self, 
                q: torch.Tensor,
                k: torch.Tensor,
                frame_size: Tuple[int, int],
                num_frames: int) -> torch.Tensor:
        """
        Compute multi-scale draft attention with learned importance weights.
        
        Args:
            q: Query tensor (B, N, D)
            k: Key tensor (B, N, D)
            frame_size: (H, W) spatial dimensions
            num_frames: Number of temporal frames
            
        Returns:
            Combined draft attention map
        """
        B, N, D = q.shape
        
        # Compute draft attention for each scale
        draft_attentions = []
        for scale in self.scales:
            draft = self._compute_scale_draft(q, k, scale, frame_size, num_frames)
            draft_attentions.append(draft)
        
        # Compute importance weights
        q_mean = q.mean(dim=1)  # B, D
        k_mean = k.mean(dim=1)  # B, D
        combined_stats = torch.cat([q_mean, k_mean], dim=-1)  # B, 2D
        importance_weights = self.importance_mlp(combined_stats)  # B, num_scales
        
        # Weighted combination
        combined_draft = torch.zeros_like(draft_attentions[0])
        for i, draft in enumerate(draft_attentions):
            weight = importance_weights[:, i:i+1, None]  # B, 1, 1
            combined_draft += weight * draft
        
        return combined_draft


class MotionAwareTemporalPooling(nn.Module):
    """Motion-Aware Temporal Pooling (MATP) component."""
    
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        
        # Simple motion estimator using frame differences
        self.motion_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        
    def _estimate_motion(self, frames: torch.Tensor) -> torch.Tensor:
        """Estimate motion between consecutive frames."""
        B, F, H, W, D = frames.shape
        
        # Reshape for motion estimation
        frames_reshaped = frames.permute(0, 4, 1, 2, 3)  # B, D, F, H, W
        
        # Compute frame differences
        frame_diff = frames_reshaped[:, :, 1:] - frames_reshaped[:, :, :-1]
        
        # Sum across channels for motion magnitude
        motion_magnitude = torch.norm(frame_diff, dim=1)  # B, F-1, H, W
        
        # Pad to maintain temporal dimension
        motion_magnitude = F.pad(motion_magnitude, (0, 0, 0, 0, 1, 0), mode='replicate')
        
        return motion_magnitude  # B, F, H, W
    
    def forward(self, 
                features: torch.Tensor,
                frame_size: Tuple[int, int],
                num_frames: int) -> torch.Tensor:
        """
        Apply motion-aware temporal pooling.
        
        Args:
            features: Feature tensor (B, N, D)
            frame_size: (H, W) spatial dimensions
            num_frames: Number of temporal frames
            
        Returns:
            Motion-weighted pooled features
        """
        B, N, D = features.shape
        H, W = frame_size
        
        # Reshape to video format
        features_video = features.view(B, num_frames, H, W, D)
        
        # Estimate motion
        motion_weights = self._estimate_motion(features_video)  # B, F, H, W
        
        # Normalize motion weights
        motion_weights = F.softmax(motion_weights.view(B, -1), dim=-1)
        motion_weights = motion_weights.view(B, num_frames, H, W)
        
        return motion_weights


class DynamicSparsityPredictor(nn.Module):
    """Dynamic Layer-wise Sparsity (DLS) predictor."""
    
    def __init__(self, dim: int, num_layers: int = 28, device: str = "cuda"):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.device = device
        
        # Layer statistics encoder
        self.stats_encoder = nn.Sequential(
            nn.Linear(5, dim // 4),  # 5 statistics features
            nn.ReLU(),
            nn.Linear(dim // 4, dim // 8),
            nn.ReLU()
        )
        
        # Sparsity predictor per layer
        self.sparsity_predictor = nn.Linear(dim // 8, 1)
        
    def _compute_layer_stats(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute layer statistics for sparsity prediction."""
        B, N, D = q.shape
        
        # Statistics: mean, std, max, min, norm
        q_stats = torch.stack([
            q.mean(dim=[1, 2]),
            q.std(dim=[1, 2]),
            q.max(dim=1)[0].max(dim=1)[0],
            q.min(dim=1)[0].min(dim=1)[0],
            q.norm(dim=-1).mean(dim=1)
        ], dim=-1)  # B, 5
        
        k_stats = torch.stack([
            k.mean(dim=[1, 2]),
            k.std(dim=[1, 2]),
            k.max(dim=1)[0].max(dim=1)[0],
            k.min(dim=1)[0].min(dim=1)[0],
            k.norm(dim=-1).mean(dim=1)
        ], dim=-1)  # B, 5
        
        # Combine statistics
        combined_stats = (q_stats + k_stats) / 2  # B, 5
        
        return combined_stats
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Predict optimal sparsity ratio for current layer.
        
        Args:
            q: Query tensor (B, N, D)
            k: Key tensor (B, N, D)
            
        Returns:
            Sparsity ratio ∈ (0, 1)
        """
        stats = self._compute_layer_stats(q, k)  # B, 5
        encoded = self.stats_encoder(stats)  # B, dim//8
        sparsity = torch.sigmoid(self.sparsity_predictor(encoded))  # B, 1
        
        # Clamp to reasonable range
        sparsity = torch.clamp(sparsity, 0.05, 0.95)
        
        return sparsity.squeeze(-1)  # B


class HybridQuantizedAttention(nn.Module):
    """Hybrid Sparse-Quantized Attention (HSQA) component."""
    
    def __init__(self, dim: int, num_bits: int = 4, device: str = "cuda"):
        super().__init__()
        self.dim = dim
        self.num_bits = num_bits
        self.device = device
        
        # Quantization parameters
        self.q_scale = nn.Parameter(torch.ones(1))
        self.k_scale = nn.Parameter(torch.ones(1))
        
    def _quantize_tensor(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to specified bit width."""
        # Simple symmetric quantization
        qmin = -(2 ** (self.num_bits - 1))
        qmax = 2 ** (self.num_bits - 1) - 1
        
        x_scaled = x / scale.clamp(min=1e-8)
        x_quant = torch.round(torch.clamp(x_scaled, qmin, qmax))
        x_dequant = x_quant * scale
        
        # Straight-through estimator for gradients
        return x + (x_dequant - x).detach()
    
    def forward(self, 
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                sparsity_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute attention with quantized Q/K and sparse pattern.
        
        Args:
            q: Query tensor (B, N, D)
            k: Key tensor (B, N, D)
            v: Value tensor (B, N, D)
            sparsity_mask: Binary mask (B, N, N)
            
        Returns:
            Attention output (B, N, D)
        """
        B, N, D = q.shape
        
        # Quantize Q and K
        q_quant = self._quantize_tensor(q, self.q_scale)
        k_quant = self._quantize_tensor(k, self.k_scale)
        
        # Compute attention with quantized Q/K
        scale_factor = 1.0 / math.sqrt(D)
        attn_scores = torch.bmm(q_quant, k_quant.transpose(-2, -1)) * scale_factor
        
        # Apply sparsity mask
        masked_scores = attn_scores.masked_fill(sparsity_mask == 0, float('-inf'))
        attn_probs = F.softmax(masked_scores, dim=-1)
        
        # Compute output
        out = torch.bmm(attn_probs, v)
        
        return out


class HADAFramework(nn.Module):
    """
    Hierarchical Adaptive Draft Attention (HADA) Framework
    Incorporates all advanced improvements: AMDA, DLS, MATP, HSQA
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 28,
        scales: List[Tuple[int, int]] = [(4, 4), (8, 8), (16, 16), (32, 32)],
        use_quantization: bool = True,
        use_motion_guidance: bool = True,
        use_dynamic_sparsity: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device
        
        # Core components
        self.multi_scale_draft = MultiScaleDraftAttention(dim, scales, device)
        self.dynamic_sparsity = DynamicSparsityPredictor(dim, num_layers, device) if use_dynamic_sparsity else None
        self.motion_guidance = MotionAwareTemporalPooling(device) if use_motion_guidance else None
        self.quantized_attention = HybridQuantizedAttention(dim, device=device) if use_quantization else None
        
        # Standard projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.scale = 1.0 / math.sqrt(dim // num_heads)
        
    def forward(self, 
                x: torch.Tensor,
                frame_size: Tuple[int, int],
                num_frames: int,
                layer_idx: int = 0,
                base_sparsity: float = 0.1) -> torch.Tensor:
        """
        Forward pass of HADA framework.
        
        Args:
            x: Input tensor (B, N, D)
            frame_size: (H, W) spatial dimensions
            num_frames: Number of temporal frames
            layer_idx: Current layer index for dynamic sparsity
            base_sparsity: Base sparsity ratio
            
        Returns:
            Output tensor (B, N, D)
        """
        B, N, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Multi-head reshape
        q = q.view(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)
        k = k.view(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)
        v = v.view(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)
        
        # Reshape for processing
        q_flat = q.reshape(B * self.num_heads, N, D // self.num_heads)
        k_flat = k.reshape(B * self.num_heads, N, D // self.num_heads)
        v_flat = v.reshape(B * self.num_heads, N, D // self.num_heads)
        
        # Compute multi-scale draft attention
        draft_attn = self.multi_scale_draft(
            q_flat[:, 0], k_flat[:, 0], frame_size, num_frames
        )
        
        # Apply motion guidance if enabled
        if self.motion_guidance is not None:
            motion_weights = self.motion_guidance(x, frame_size, num_frames)
            # Incorporate motion weights into draft attention
            motion_weights = motion_weights.view(B, -1)
            draft_attn = draft_attn * motion_weights.unsqueeze(-1)
        
        # Predict dynamic sparsity
        if self.dynamic_sparsity is not None:
            sparsity_ratio = self.dynamic_sparsity(q_flat[:, 0], k_flat[:, 0])
            sparsity_ratio = sparsity_ratio.mean()  # Average across batch
        else:
            sparsity_ratio = base_sparsity
        
        # Generate sparsity mask
        k_sparse = int(draft_attn.shape[-1] * sparsity_ratio)
        _, top_indices = torch.topk(draft_attn.view(B, -1), k_sparse, dim=-1)
        
        mask_flat = torch.zeros_like(draft_attn.view(B, -1))
        mask_flat.scatter_(-1, top_indices, 1.0)
        mask = mask_flat.view(B, draft_attn.shape[1], draft_attn.shape[2])
        
        # Lift mask to full resolution
        scale_factor = int(math.sqrt(N / mask.shape[1]))
        mask_full = mask.repeat_interleave(scale_factor, dim=1).repeat_interleave(scale_factor, dim=2)
        mask_full = mask_full.unsqueeze(1).expand(B, self.num_heads, N, N)
        mask_full = mask_full.reshape(B * self.num_heads, N, N)
        
        # Compute attention
        if self.quantized_attention is not None:
            # Use hybrid quantized attention
            q_flat_quant = q_flat.view(B * self.num_heads, N, D // self.num_heads)
            k_flat_quant = k_flat.view(B * self.num_heads, N, D // self.num_heads)
            v_flat_quant = v_flat.view(B * self.num_heads, N, D // self.num_heads)
            
            out_flat = self.quantized_attention(q_flat_quant, k_flat_quant, v_flat_quant, mask_full)
        else:
            # Standard sparse attention
            attn_scores = torch.bmm(q_flat, k_flat.transpose(-2, -1)) * self.scale
            masked_scores = attn_scores.masked_fill(mask_full == 0, float('-inf'))
            attn_probs = F.softmax(masked_scores, dim=-1)
            out_flat = torch.bmm(attn_probs, v_flat)
        
        # Reshape output
        out = out_flat.view(B, self.num_heads, N, D // self.num_heads).transpose(1, 2)
        out = out.reshape(B, N, D)
        
        # Output projection
        out = self.out_proj(out)
        
        return out
    
    def load_weights(self, checkpoint_path: str):
        """Load pre-trained weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint)
        
    def save_weights(self, checkpoint_path: str):
        """Save model weights to checkpoint."""
        torch.save(self.state_dict(), checkpoint_path)


class HADAConfig:
    """Configuration class for HADA framework."""
    
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 28,
        scales: List[Tuple[int, int]] = [(4, 4), (8, 8), (16, 16), (32, 32)],
        use_quantization: bool = True,
        use_motion_guidance: bool = True,
        use_dynamic_sparsity: bool = True
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.scales = scales
        self.use_quantization = use_quantization
        self.use_motion_guidance = use_motion_guidance
        self.use_dynamic_sparsity = use_dynamic_sparsity


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N, D = 2, 128 * 16 * 9, 768  # 9 frames, 128x16 patches
    frame_size = (128, 16 * 16)  # H, W
    num_frames = 9
    
    # Initialize HADA framework
    hada = HADAFramework(
        dim=D,
        num_heads=12,
        num_layers=28,
        use_quantization=True,
        use_motion_guidance=True,
        use_dynamic_sparsity=True,
        device=device
    ).to(device)
    
    # Create dummy input
    x = torch.randn(B, N, D, device=device)
    
    # Forward pass
    with torch.no_grad():
        output = hada(x, frame_size, num_frames, layer_idx=14)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("HADA framework test passed!")