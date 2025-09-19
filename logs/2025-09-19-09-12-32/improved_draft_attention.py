"""
Improved DraftAttention with adaptive features as proposed in the paper.
Includes dynamic kernel selection, sparsity scheduling, quantization, and distributed support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from draft_attention import DraftAttention, DraftAttentionConfig


class AdaptiveDraftAttention(DraftAttention):
    """
    Advanced version of DraftAttention with adaptive kernel selection and dynamic sparsity.
    
    New features:
    1. Content-aware kernel size selection
    2. Step-wise sparsity adjustment
    3. Multi-scale draft attention
    4. Temporal-separate pooling
    5. Quantization support
    6. Distributed inference
    """
    
    def __init__(
        self,
        base_sparsity: float = 0.75,
        min_sparsity: float = 0.5,
        max_sparsity: float = 0.95,
        adaptive_kernels: List[Tuple[int, int]] = [(4, 8), (8, 16), (16, 32)],
        use_quantization: bool = False,
        quantization_bits: int = 8,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        **kwargs
    ):
        super().__init__(sparsity_ratio=base_sparsity, **kwargs)
        
        self.base_sparsity = base_sparsity
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        self.adaptive_kernels = adaptive_kernels
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        self.distributed = distributed
        self.world_size = world_size
        self.rank = rank
        
        # Content complexity analyzer
        self.complexity_analyzer = nn.AdaptiveAvgPool1d(1)  # Simple complexity measure
        
        # Multi-scale draft attention weights
        self.scale_weights = nn.Parameter(torch.ones(len(adaptive_kernels)))
        
        # Quantization parameters
        if self.use_quantization:
            self.register_buffer('quant_scale', torch.tensor(1.0))
            self.register_buffer('quant_zero_point', torch.tensor(0))
    
    def _compute_content_complexity(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute local complexity for adaptive kernel selection.
        
        Args:
            tensor: [B, n_heads, seq_len, d_head]
            
        Returns:
            complexity: [B, n_heads, seq_len] complexity scores
        """
        # Compute gradient-based complexity
        B, n_heads, seq_len, d_head = tensor.shape
        
        # Reshape for complexity analysis
        tensor_reshaped = tensor.view(-1, d_head, seq_len)
        
        # Compute local variance as complexity measure
        local_mean = F.avg_pool1d(
            tensor_reshaped.mean(dim=1, keepdim=True),
            kernel_size=3, stride=1, padding=1
        )
        local_var = ((tensor_reshaped.mean(dim=1, keepdim=True) - local_mean) ** 2).mean(dim=-1)
        
        complexity = local_var.view(B, n_heads, -1)
        return complexity
    
    def _select_adaptive_kernel(
        self, 
        query: torch.Tensor,
        key: torch.Tensor,
        complexity: torch.Tensor
    ) -> Tuple[int, int]:
        """
        Select optimal pooling kernel based on content complexity.
        
        Args:
            query: [B, n_heads, seq_len, d_head]
            key: [B, n_heads, seq_len, d_head]
            complexity: [B, n_heads, seq_len] complexity scores
            
        Returns:
            selected_kernel: (temporal, spatial) kernel size
        """
        # Compute global complexity score
        global_complexity = complexity.mean(dim=[1, 2])  # [B]
        
        # Map complexity to kernel index (0 = small, 2 = large)
        normalized_complexity = (global_complexity - global_complexity.min()) / \
                               (global_complexity.max() - global_complexity.min() + 1e-8)
        kernel_idx = (normalized_complexity * (len(self.adaptive_kernels) - 1)).long()
        
        # Select kernel for each batch element
        kernels = [self.adaptive_kernels[idx.item()] for idx in kernel_idx]
        
        # For simplicity, use majority vote or first element
        selected_kernel = kernels[0]  # In practice, handle per-batch
        
        return selected_kernel
    
    def _compute_dynamic_sparsity(self, step: int, total_steps: int) -> float:
        """
        Compute sparsity ratio based on denoising step.
        
        Args:
            step: Current denoising step
            total_steps: Total denoising steps
            
        Returns:
            sparsity_ratio: Dynamic sparsity for current step
        """
        # Use cosine schedule for sparsity
        progress = step / total_steps
        sparsity_range = self.max_sparsity - self.min_sparsity
        
        # Higher sparsity in early steps, lower in final steps
        dynamic_sparsity = self.max_sparsity - sparsity_range * (1 - np.cos(progress * np.pi)) / 2
        
        return max(self.min_sparsity, min(self.max_sparsity, dynamic_sparsity))
    
    def _compute_multi_scale_draft(
        self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-scale draft attention with learned combination.
        
        Args:
            query: [B, n_heads, seq_len, d_head]
            key: [B, n_heads, seq_len, d_head]
            value: [B, n_heads, seq_len, d_head]
            
        Returns:
            combined_draft: Weighted combination of multi-scale drafts
        """
        B, n_heads, seq_len, d_head = query.shape
        
        draft_maps = []
        for kernel in self.adaptive_kernels:
            # Temporarily set kernel
            original_kernel = self.pooling_kernel
            self.pooling_kernel = kernel
            
            try:
                draft_attention, _ = self._compute_draft_attention(query, key, value)
                draft_maps.append(draft_attention)
            finally:
                self.pooling_kernel = original_kernel
        
        # Stack and combine with learned weights
        draft_stack = torch.stack(draft_maps, dim=0)  # [n_kernels, B, n_heads, g, g]
        weights = F.softmax(self.scale_weights, dim=0).view(-1, 1, 1, 1, 1)
        
        combined_draft = (weights * draft_stack).sum(dim=0)
        return combined_draft
    
    def _apply_quantization(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply quantization to attention weights.
        
        Args:
            tensor: Input tensor to quantize
            
        Returns:
            quantized_tensor: Quantized tensor
        """
        if not self.use_quantization:
            return tensor
            
        if self.quantization_bits == 8:
            # INT8 quantization
            qmin, qmax = -128, 127
        elif self.quantization_bits == 4:
            # INT4 quantization
            qmin, qmax = -8, 7
        else:
            return tensor
            
        # Compute scale and zero point
        min_val, max_val = tensor.min(), tensor.max()
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)
        
        # Dequantize for computation
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
    
    def _distributed_draft_attention(
        self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute draft attention in distributed setting.
        
        Args:
            query: [B, n_heads, seq_len, d_head]
            key: [B, n_heads, seq_len, d_head]
            value: [B, n_heads, seq_len, d_head]
            
        Returns:
            draft_attention: Synchronized draft attention across GPUs
        """
        if not self.distributed:
            draft_attention, _ = self._compute_draft_attention(query, key, value)
            return draft_attention
        
        # Split across heads for distributed computation
        heads_per_gpu = query.shape[1] // self.world_size
        start_head = self.rank * heads_per_gpu
        end_head = (self.rank + 1) * heads_per_gpu
        
        # Compute local draft attention
        local_query = query[:, start_head:end_head, :, :]
        local_key = key[:, start_head:end_head, :, :]
        local_value = value[:, start_head:end_head, :, :]
        
        local_draft, _ = self._compute_draft_attention(local_query, local_key, local_value)
        
        # All-reduce draft attention across GPUs
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(local_draft)
        
        return local_draft
    
    def _temporal_separate_pooling(
        self, 
        tensor: torch.Tensor,
        temporal_kernel: int = 8,
        spatial_kernel: int = 16
    ) -> torch.Tensor:
        """
        Apply different pooling for temporal vs spatial dimensions.
        
        Args:
            tensor: [B, n_heads, seq_len, d_head]
            temporal_kernel: Temporal pooling size
            spatial_kernel: Spatial pooling size
            
        Returns:
            pooled_tensor: Pooled tensor with separate temporal/spatial reduction
        """
        B, n_heads, seq_len, d_head = tensor.shape
        
        # Infer dimensions
        T = seq_len // (spatial_kernel * spatial_kernel)
        H = W = int(np.sqrt(seq_len // T))
        
        # Reshape for separate pooling
        tensor_reshaped = tensor.view(B, n_heads, T, H, W, d_head)
        
        # Temporal pooling
        temp_pooled = F.avg_pool3d(
            tensor_reshaped.permute(0, 1, 5, 2, 3, 4).reshape(-1, d_head, T, H, W),
            kernel_size=(temporal_kernel, 1, 1),
            stride=(temporal_kernel, 1, 1)
        ).view(B, n_heads, d_head, -1, H, W)
        
        # Spatial pooling
        spatial_pooled = F.avg_pool3d(
            temp_pooled,
            kernel_size=(1, spatial_kernel, spatial_kernel),
            stride=(1, spatial_kernel, spatial_kernel)
        )
        
        return spatial_pooled.permute(0, 1, 3, 4, 5, 2).reshape(B, n_heads, -1, d_head)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        step: int = 0,
        total_steps: int = 50,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with adaptive features.
        
        Args:
            query: [B, n_heads, seq_len, d_head]
            key: [B, n_heads, seq_len, d_head]
            value: [B, n_heads, seq_len, d_head]
            step: Current denoising step
            total_steps: Total denoising steps
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing output and metadata
        """
        B, n_heads, seq_len, d_head = query.shape
        
        # Use dense attention for initial fallback steps
        if step < self.fallback_steps:
            return super().forward(query, key, value, step, return_attention)
        
        # Compute content complexity for adaptive decisions
        complexity = self._compute_content_complexity(query)
        
        # Select adaptive kernel
        selected_kernel = self._select_adaptive_kernel(query, key, complexity)
        original_kernel = self.pooling_kernel
        self.pooling_kernel = selected_kernel
        
        try:
            # Compute dynamic sparsity
            dynamic_sparsity = self._compute_dynamic_sparsity(step, total_steps)
            
            # Multi-scale draft attention
            if len(self.adaptive_kernels) > 1:
                draft_attention = self._compute_multi_scale_draft(query, key, value)
            else:
                draft_attention, _ = self._compute_draft_attention(query, key, value)
            
            # Apply quantization if enabled
            draft_attention = self._apply_quantization(draft_attention)
            
            # Create sparsity mask with dynamic ratio
            mask = self._create_sparsity_mask(draft_attention, dynamic_sparsity)
            
            # Distributed computation
            if self.distributed:
                draft_attention = self._distributed_draft_attention(query, key, value)
            
            # Reorder and compute sparse attention
            reordered_query, query_indices = self._reorder_tokens(query, mask)
            reordered_key, key_indices = self._reorder_tokens(key, mask)
            reordered_value, value_indices = self._reorder_tokens(value, mask)
            
            # Apply quantization to attention computation
            reordered_query = self._apply_quantization(reordered_query)
            reordered_key = self._apply_quantization(reordered_key)
            reordered_value = self._apply_quantization(reordered_value)
            
            # Compute sparse attention
            scale = 1.0 / np.sqrt(d_head)
            sparse_attention = torch.matmul(reordered_query, reordered_key.transpose(-2, -1)) * scale
            sparse_attention = F.softmax(sparse_attention, dim=-1)
            sparse_attention = self._apply_quantization(sparse_attention)
            
            sparse_output = torch.matmul(sparse_attention, reordered_value)
            sparse_output = self._apply_quantization(sparse_output)
            
            # Restore original order
            output = self._restore_order(sparse_output, value_indices, (B, n_heads, seq_len, d_head))
            
            result = {
                "output": output,
                "metadata": {
                    "selected_kernel": selected_kernel,
                    "dynamic_sparsity": dynamic_sparsity,
                    "complexity_score": complexity.mean().item(),
                    "quantization_enabled": self.use_quantization,
                    "distributed": self.distributed
                }
            }
            
            if return_attention:
                result["attention"] = sparse_attention
            
            return result
            
        finally:
            # Restore original kernel
            self.pooling_kernel = original_kernel
    
    def get_efficiency_stats(self) -> Dict[str, float]:
        """Get efficiency statistics for the current configuration."""
        stats = {
            "base_sparsity": self.base_sparsity,
            "min_sparsity": self.min_sparsity,
            "max_sparsity": self.max_sparsity,
            "adaptive_kernels": len(self.adaptive_kernels),
            "quantization_bits": self.quantization_bits if self.use_quantization else 32,
            "distributed_world_size": self.world_size if self.distributed else 1,
            "expected_speedup": self._estimate_speedup()
        }
        return stats
    
    def _estimate_speedup(self) -> float:
        """Estimate theoretical speedup based on configuration."""
        base_reduction = 128  # From 8×16 pooling
        sparsity_factor = 1.0 / (1.0 - self.base_sparsity)
        quantization_factor = 4.0 if self.use_quantization and self.quantization_bits == 4 else 1.0
        
        estimated_speedup = base_reduction * sparsity_factor / quantization_factor
        return min(estimated_speedup, 2.0)  # Cap at reasonable maximum


class AdaptiveDraftAttentionConfig(DraftAttentionConfig):
    """Configuration for AdaptiveDraftAttention."""
    
    def __init__(
        self,
        base_sparsity: float = 0.75,
        min_sparsity: float = 0.5,
        max_sparsity: float = 0.95,
        adaptive_kernels: List[Tuple[int, int]] = [(4, 8), (8, 16), (16, 32)],
        use_quantization: bool = False,
        quantization_bits: int = 8,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_sparsity = base_sparsity
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        self.adaptive_kernels = adaptive_kernels
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        self.distributed = distributed
        self.world_size = world_size
        self.rank = rank
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'base_sparsity': self.base_sparsity,
            'min_sparsity': self.min_sparsity,
            'max_sparsity': self.max_sparsity,
            'adaptive_kernels': self.adaptive_kernels,
            'use_quantization': self.use_quantization,
            'quantization_bits': self.quantization_bits,
            'distributed': self.distributed,
            'world_size': self.world_size,
            'rank': self.rank
        })
        return base_dict