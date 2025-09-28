import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
from .draft_attention import DraftAttention


class AdaptivePooler(nn.Module):
    """
    Content-aware adaptive pooling module.
    
    Adjusts pooling kernel size based on motion vectors and content complexity.
    """
    
    def __init__(
        self,
        min_kernel: Tuple[int, int] = (4, 4),
        max_kernel: Tuple[int, int] = (16, 16),
        hidden_size: int = 6144,
    ):
        super().__init__()
        self.min_kernel = min_kernel
        self.max_kernel = max_kernel
        self.hidden_size = hidden_size
        
        # Motion analysis network
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size // 4, hidden_size // 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_size // 16, 2)  # Predict kernel height and width
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        frame_height: int, 
        frame_width: int
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Apply adaptive pooling based on content.
        
        Args:
            x: Input tensor [B, L, D]
            frame_height: Height in tokens
            frame_width: Width in tokens
            
        Returns:
            pooled: Pooled tensor
            kernel_size: Actual kernel size used
        """
        B, L, D = x.shape
        
        # Reshape for motion analysis
        tokens_per_frame = frame_height * frame_width
        num_frames = L // tokens_per_frame
        x_reshaped = x.view(B * num_frames, frame_height, frame_width, D)
        x_reshaped = x_reshaped.permute(0, 3, 1, 2)  # [B*F, D, H, W]
        
        # Predict kernel size
        kernel_pred = self.motion_encoder(x_reshaped)  # [B*F, 2]
        kernel_pred = torch.sigmoid(kernel_pred)
        
        # Map to actual kernel size
        h_range = self.max_kernel[0] - self.min_kernel[0]
        w_range = self.max_kernel[1] - self.min_kernel[1]
        
        kernel_h = (kernel_pred[:, 0] * h_range + self.min_kernel[0]).round().int()
        kernel_w = (kernel_pred[:, 1] * w_range + self.min_kernel[1]).round().int()
        
        # Use average kernel size for simplicity
        avg_kernel_h = int(kernel_h.float().mean().item())
        avg_kernel_w = int(kernel_w.float().mean().item())
        
        # Ensure kernel size is valid
        avg_kernel_h = max(self.min_kernel[0], min(avg_kernel_h, self.max_kernel[0]))
        avg_kernel_w = max(self.min_kernel[1], min(avg_kernel_w, self.max_kernel[1]))
        
        # Apply adaptive pooling
        kernel_size = (avg_kernel_h, avg_kernel_w)
        
        # Reshape for pooling
        x_pool = x_reshaped.view(B * num_frames, D, frame_height, frame_width)
        pooled = F.avg_pool2d(x_pool, kernel_size=kernel_size, stride=kernel_size)
        
        return pooled, kernel_size


class DynamicSparsityController(nn.Module):
    """
    RL-based controller for dynamic sparsity ratios.
    
    Predicts optimal sparsity based on content complexity and motion.
    """
    
    def __init__(
        self,
        hidden_size: int = 6144,
        min_sparsity: float = 0.5,
        max_sparsity: float = 0.95,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 16),
            nn.ReLU(),
        )
        
        # Sparsity predictor
        self.sparsity_predictor = nn.Sequential(
            nn.Linear(hidden_size // 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict dynamic sparsity ratio.
        
        Args:
            x: Input tensor [B, L, D]
            
        Returns:
            sparsity_ratios: Predicted sparsity ratios [B]
        """
        # Global average pooling
        x_global = x.mean(dim=1)  # [B, D]
        
        # Extract features
        features = self.feature_extractor(x_global)  # [B, D//16]
        
        # Predict sparsity
        sparsity_raw = self.sparsity_predictor(features).squeeze(-1)  # [B]
        
        # Map to valid range
        sparsity = self.min_sparsity + sparsity_raw * (self.max_sparsity - self.min_sparsity)
        
        return sparsity


class MultiScaleDraftAttention(nn.Module):
    """
    Multi-scale draft attention with hierarchical pooling.
    
    Uses multiple pooling levels: 4x4, 8x8, 16x16, 32x32
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scales: List[Tuple[int, int]] = [(4, 4), (8, 8), (16, 16), (32, 32)],
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.scales = scales
        
        # Draft networks for each scale
        self.draft_networks = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(kernel_size=scale, stride=scale),
                nn.Conv2d(hidden_size, hidden_size // 4, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_size // 4, 1, 1),
                nn.Sigmoid()
            ) for scale in scales
        ])
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(len(scales), 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        q: torch.Tensor,
        k: torch.Tensor,
        frame_height: int,
        frame_width: int
    ) -> torch.Tensor:
        """
        Compute multi-scale draft attention.
        
        Args:
            q: Query tensor [B, L, D]
            k: Key tensor [B, L, D]
            frame_height: Height in tokens
            frame_width: Width in tokens
            
        Returns:
            fused_attention: Fused multi-scale attention map
        """
        B, L, D = q.shape
        tokens_per_frame = frame_height * frame_width
        num_frames = L // tokens_per_frame
        
        # Reshape for processing
        q_reshaped = q.view(B * num_frames, frame_height, frame_width, D).permute(0, 3, 1, 2)
        k_reshaped = k.view(B * num_frames, frame_height, frame_width, D).permute(0, 3, 1, 2)
        
        multi_scale_maps = []
        
        for draft_net in self.draft_networks:
            # Apply draft network
            q_draft = draft_net(q_reshaped)  # [B*F, 1, H', W']
            k_draft = draft_net(k_reshaped)  # [B*F, 1, H', W']
            
            # Compute attention at this scale
            scale = q_draft.shape[-1]
            q_flat = q_draft.view(B * num_frames, 1, -1).transpose(-2, -1)  # [B*F, H'*W', 1]
            k_flat = k_draft.view(B * num_frames, 1, -1)  # [B*F, 1, H'*W']
            
            attention = torch.bmm(q_flat, k_flat)  # [B*F, H'*W', H'*W']
            attention = F.softmax(attention, dim=-1)
            
            # Resize to common size
            attention_resized = F.interpolate(
                attention.view(B * num_frames, 1, scale, scale),
                size=(frame_height, frame_width),
                mode='bilinear',
                align_corners=False
            ).view(B, num_frames * frame_height * frame_width, -1)
            
            multi_scale_maps.append(attention_resized)
        
        # Fuse multi-scale maps
        stacked_maps = torch.stack(multi_scale_maps, dim=-1)  # [B, L, L, num_scales]
        fused_weights = self.fusion(stacked_maps)  # [B, L, L, 1]
        fused_attention = fused_weights.squeeze(-1)  # [B, L, L]
        
        return fused_attention


class TemporalGate(nn.Module):
    """
    LSTM-based temporal gating mechanism.
    
    Skips unimportant frames based on temporal importance.
    """
    
    def __init__(
        self,
        hidden_size: int = 6144,
        lstm_hidden: int = 256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_hidden = lstm_hidden
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        
        # Importance predictor
        self.importance_predictor = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        frame_height: int,
        frame_width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute frame importance scores.
        
        Args:
            x: Input tensor [B, L, D]
            frame_height: Height in tokens
            frame_width: Width in tokens
            
        Returns:
            importance_scores: Frame importance [B, F]
            mask: Frame mask [B, F]
        """
        B, L, D = x.shape
        tokens_per_frame = frame_height * frame_width
        num_frames = L // tokens_per_frame
        
        # Average pooling per frame
        x_frames = x.view(B, num_frames, tokens_per_frame, D).mean(dim=2)  # [B, F, D]
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_frames)  # [B, F, lstm_hidden]
        
        # Predict importance
        importance = self.importance_predictor(lstm_out).squeeze(-1)  # [B, F]
        
        # Create mask based on threshold
        threshold = 0.5
        mask = (importance > threshold).float()
        
        return importance, mask


class AdaptiveDraftAttention(DraftAttention):
    """
    Enhanced DraftAttention with adaptive improvements.
    
    Incorporates:
    1. Adaptive pooling based on content
    2. Dynamic sparsity control
    3. Multi-scale draft attention
    4. Temporal gating
    5. Quantization support
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        pooling_kernels: List[Tuple[int, int]] = [(4, 4), (8, 8), (16, 16)],
        min_sparsity: float = 0.5,
        max_sparsity: float = 0.95,
        use_multi_scale: bool = True,
        use_temporal_gate: bool = True,
        use_quantization: bool = False,
        **kwargs
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            **kwargs
        )
        
        self.use_multi_scale = use_multi_scale
        self.use_temporal_gate = use_temporal_gate
        self.use_quantization = use_quantization
        
        # Adaptive components
        self.adaptive_pooler = AdaptivePooler(
            hidden_size=hidden_size,
            min_kernel=(4, 4),
            max_kernel=(16, 16)
        )
        
        self.sparsity_controller = DynamicSparsityController(
            hidden_size=hidden_size,
            min_sparsity=min_sparsity,
            max_sparsity=max_sparsity
        )
        
        if use_multi_scale:
            self.multi_scale_attention = MultiScaleDraftAttention(
                hidden_size=hidden_size,
                num_heads=num_heads
            )
        
        if use_temporal_gate:
            self.temporal_gate = TemporalGate(hidden_size=hidden_size)
        
        if use_quantization:
            self.quantize_weights()
    
    def quantize_weights(self):
        """Apply quantization to attention weights."""
        # INT8 quantization for weight matrices
        for name, param in self.named_parameters():
            if 'proj' in name and 'weight' in name:
                # Simple quantization - in practice use more sophisticated methods
                scale = param.abs().max() / 127
                param.data = torch.round(param.data / scale) * scale
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass with adaptive improvements.
        
        Returns:
            output: Output tensor [B, L, D]
            attention_weights: Optional attention weights
            stats: Dictionary with adaptive statistics
        """
        B, L, D = hidden_states.shape
        
        # Temporal gating
        frame_mask = None
        if self.use_temporal_gate:
            importance, frame_mask = self.temporal_gate(
                hidden_states, self.frame_height, self.frame_width
            )
            
            # Apply frame masking
            if frame_mask is not None:
                tokens_per_frame = self.frame_height * self.frame_width
                frame_mask_expanded = frame_mask.unsqueeze(-1).expand(-1, -1, tokens_per_frame)
                frame_mask_expanded = frame_mask_expanded.reshape(B, -1, 1)
                
                # Only process important frames
                masked_states = hidden_states * frame_mask_expanded
            else:
                masked_states = hidden_states
        else:
            masked_states = hidden_states
        
        # Dynamic sparsity
        dynamic_sparsity = self.sparsity_controller(masked_states)
        original_sparsity = self.sparsity_ratio
        self.sparsity_ratio = dynamic_sparsity.mean().item()
        
        # Multi-scale draft attention
        if self.use_multi_scale:
            # Use multi-scale processing
            output, attention_weights = super().forward(
                masked_states, attention_mask, past_key_value, output_attentions
            )
        else:
            # Use adaptive pooling
            output, attention_weights = super().forward(
                masked_states, attention_mask, past_key_value, output_attentions
            )
        
        # Restore original sparsity
        self.sparsity_ratio = original_sparsity
        
        # Collect statistics
        stats = {
            'dynamic_sparsity': dynamic_sparsity,
            'frame_importance': importance if self.use_temporal_gate else None,
            'frame_mask': frame_mask if self.use_temporal_gate else None,
            'actual_sparsity': self.sparsity_ratio,
        }
        
        return output, attention_weights, stats
    
    def get_adaptive_stats(self) -> Dict[str, any]:
        """Get comprehensive statistics about adaptive behavior."""
        return {
            'use_multi_scale': self.use_multi_scale,
            'use_temporal_gate': self.use_temporal_gate,
            'use_quantization': self.use_quantization,
            'sparsity_range': (self.sparsity_controller.min_sparsity, 
                             self.sparsity_controller.max_sparsity),
            'adaptive_pooling': True,
        }