"""
Improved XAttention implementation with proposed enhancements.
Includes adaptive warmup, multi-scale patterns, content-adaptive sparsity, and gradient optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import math


class ContentAnalyzer(nn.Module):
    """Lightweight content analyzer for adaptive strategies."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # warmup_steps, density, complexity
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze content complexity.
        
        Args:
            x: [B, L, D] input tensor
            
        Returns:
            dict with predicted parameters
        """
        # Global average pooling
        pooled = x.mean(dim=1)  # [B, D]
        
        # Predict parameters
        params = self.net(pooled)  # [B, 3]
        
        warmup_steps = torch.sigmoid(params[:, 0]) * 5  # [0, 5]
        density = torch.sigmoid(params[:, 1]) * 0.3 + 0.05  # [0.05, 0.35]
        complexity = torch.sigmoid(params[:, 2])  # [0, 1]
        
        return {
            'warmup_steps': warmup_steps,
            'density': density,
            'complexity': complexity
        }


class MultiScalePattern(nn.Module):
    """Multi-scale antidiagonal pattern computation."""
    
    def __init__(self, strides: List[int] = [4, 8, 16]):
        super().__init__()
        self.strides = strides
        
        # Learnable weights for combining scales
        self.scale_weights = nn.Parameter(torch.ones(len(strides)))
        
    def forward(self, attn_block: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale antidiagonal scores.
        
        Args:
            attn_block: [B, B] attention values for a block
            
        Returns:
            combined_score: scalar importance score
        """
        B = attn_block.size(0)
        combined_scores = []
        
        for i, stride in enumerate(self.strides):
            scores = []
            
            # Sample along antidiagonals with given stride
            for k in range(0, 2 * B - 1, stride):
                elements = []
                for j in range(B):
                    l = k - j
                    if 0 <= l < B:
                        elements.append(attn_block[j, l])
                
                if elements:
                    scores.append(torch.stack(elements).sum())
            
            if scores:
                scale_score = torch.stack(scores).sum()
            else:
                scale_score = torch.tensor(0.0, device=attn_block.device)
            
            combined_scores.append(scale_score * self.scale_weights[i])
        
        return torch.stack(combined_scores).sum()


class GradientThresholdOptimizer(nn.Module):
    """Gradient-based threshold optimization."""
    
    def __init__(self, num_heads: int, learning_rate: float = 0.01):
        super().__init__()
        self.num_heads = num_heads
        self.learning_rate = learning_rate
        
        # Threshold parameters with gradient support
        self.thresholds = nn.Parameter(torch.ones(num_heads) * 0.9)
        
        # Momentum terms
        self.register_buffer('momentum', torch.zeros(num_heads))
        self.register_buffer('velocity', torch.zeros(num_heads))
        
    def forward(self, head_idx: int) -> float:
        """Get threshold for specific head."""
        return torch.sigmoid(self.thresholds[head_idx]).item()
    
    def update_thresholds(self, gradients: torch.Tensor):
        """Update thresholds using Adam optimizer."""
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        # Update momentum
        self.momentum = beta1 * self.momentum + (1 - beta1) * gradients
        
        # Update velocity
        self.velocity = beta2 * self.velocity + (1 - beta2) * (gradients ** 2)
        
        # Bias correction
        momentum_corr = self.momentum / (1 - beta1)
        velocity_corr = self.velocity / (1 - beta2)
        
        # Update thresholds
        with torch.no_grad():
            self.thresholds -= self.learning_rate * momentum_corr / (torch.sqrt(velocity_corr) + eps)
            self.thresholds.clamp_(0.1, 0.95)  # Keep thresholds in reasonable range


class AdaptiveBlockSizer(nn.Module):
    """Dynamic block sizing based on head characteristics."""
    
    def __init__(self, num_heads: int, block_sizes: List[int] = [4, 8, 16, 32]):
        super().__init__()
        self.num_heads = num_heads
        self.block_sizes = block_sizes
        
        # Head-specific block size selection
        self.block_selectors = nn.Parameter(torch.randn(num_heads, len(block_sizes)))
        
    def get_block_size(self, head_idx: int) -> int:
        """Get optimal block size for given head."""
        probs = F.softmax(self.block_selectors[head_idx], dim=0)
        selected_idx = torch.multinomial(probs, 1).item()
        return self.block_sizes[selected_idx]


class XAttentionImproved(nn.Module):
    """
    Improved XAttention with proposed enhancements:
    1. Adaptive warmup strategy
    2. Multi-scale antidiagonal patterns
    3. Content-adaptive sparsity
    4. Gradient-based threshold optimization
    5. Dynamic block sizing
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        default_block_size: int = 8,
        strides: List[int] = [4, 8, 16],
        default_threshold: float = 0.9,
        max_seq_len: int = 8192,
        use_adaptive_warmup: bool = True,
        use_multi_scale: bool = True,
        use_content_adaptive: bool = True,
        use_gradient_optimization: bool = True,
        use_dynamic_blocks: bool = True,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.default_block_size = default_block_size
        self.strides = strides
        self.default_threshold = default_threshold
        self.max_seq_len = max_seq_len
        
        # Feature flags
        self.use_adaptive_warmup = use_adaptive_warmup
        self.use_multi_scale = use_multi_scale
        self.use_content_adaptive = use_content_adaptive
        self.use_gradient_optimization = use_gradient_optimization
        self.use_dynamic_blocks = use_dynamic_blocks
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Enhanced components
        if use_content_adaptive:
            self.content_analyzer = ContentAnalyzer(dim)
        
        if use_multi_scale:
            self.multi_scale = MultiScalePattern(strides)
        
        if use_gradient_optimization:
            self.threshold_optimizer = GradientThresholdOptimizer(num_heads)
        else:
            self.thresholds = nn.Parameter(torch.ones(num_heads) * default_threshold)
        
        if use_dynamic_blocks:
            self.block_sizer = AdaptiveBlockSizer(num_heads)
        
        # Warmup counter
        self.register_buffer('warmup_steps', torch.zeros(1))
        self.register_buffer('max_warmup', torch.tensor(5.0))
        
    def _compute_adaptive_parameters(self, x: torch.Tensor) -> Dict[str, Any]:
        """Compute adaptive parameters based on content."""
        params = {}
        
        if self.use_content_adaptive:
            content_params = self.content_analyzer(x)
            params.update(content_params)
        else:
            B = x.size(0)
            params.update({
                'warmup_steps': torch.ones(B) * 5.0,
                'density': torch.ones(B) * 0.1,
                'complexity': torch.ones(B) * 0.5
            })
        
        return params
    
    def _compute_block_scores(
        self, 
        attn_block: torch.Tensor, 
        use_multi_scale: bool = True
    ) -> torch.Tensor:
        """
        Compute block importance scores with optional multi-scale patterns.
        
        Args:
            attn_block: [B, B] attention values for a block
            use_multi_scale: whether to use multi-scale patterns
            
        Returns:
            score: scalar importance score
        """
        if use_multi_scale and self.use_multi_scale:
            return self.multi_scale(attn_block)
        else:
            # Fallback to single-scale antidiagonal scoring
            return self._single_scale_antidiagonal(attn_block)
    
    def _single_scale_antidiagonal(self, attn_block: torch.Tensor, stride: int = 8) -> torch.Tensor:
        """Single-scale antidiagonal scoring."""
        B = attn_block.size(0)
        scores = []
        
        for k in range(0, 2 * B - 1, stride):
            elements = []
            for i in range(B):
                j = k - i
                if 0 <= j < B:
                    elements.append(attn_block[i, j])
            
            if elements:
                scores.append(torch.stack(elements).sum())
        
        if not scores:
            return torch.tensor(0.0, device=attn_block.device)
        
        return torch.stack(scores).sum()
    
    def _select_blocks_adaptive(
        self, 
        attn_approx: torch.Tensor, 
        threshold: float, 
        target_density: float = 0.1,
        block_size: int = 8
    ) -> torch.Tensor:
        """
        Select blocks with adaptive density control.
        
        Args:
            attn_approx: [NB, B, B] approximate attention map
            threshold: selection threshold
            target_density: desired density
            block_size: block size
            
        Returns:
            mask: [L, L] binary mask
        """
        L = attn_approx.size(0) * block_size
        NB = attn_approx.size(0)
        
        # Compute scores
        scores = []
        for b in range(NB):
            score = self._compute_block_scores(attn_approx[b])
            scores.append(score)
        
        scores = torch.stack(scores)
        scores = F.softmax(scores, dim=0)
        
        # Adaptive selection based on target density
        num_blocks = max(1, int(target_density * NB))
        _, selected_indices = torch.topk(scores, num_blocks)
        
        # Create mask
        mask = torch.zeros(L, L, dtype=torch.bool, device=attn_approx.device)
        for idx in selected_indices:
            start = idx * block_size
            end = (idx + 1) * block_size
            mask[start:end, start:end] = True
        
        return mask
    
    def _approximate_attention_map(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor,
        block_size: int
    ) -> torch.Tensor:
        """Create approximate attention map."""
        L, d = q.size()
        NB = L // block_size
        
        attn_approx = []
        
        for b in range(NB):
            q_slice = q[b * block_size:(b + 1) * block_size]
            k_slice = k[b * block_size:(b + 1) * block_size]
            
            # Use appropriate stride based on block size
            stride = min(8, block_size // 2)
            
            # Reshape and compute
            q_reshaped = self._reshape_along_antidiagonal(q_slice, stride)
            k_reshaped = self._reshape_along_antidiagonal(k_slice, stride)
            
            scale = 1.0 / np.sqrt(self.head_dim * stride)
            attn = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            
            attn_block = attn.mean(dim=0)
            attn_approx.append(attn_block)
        
        return torch.stack(attn_approx)
    
    def _reshape_along_antidiagonal(self, x: torch.Tensor, stride: int) -> torch.Tensor:
        """Reshape tensor along antidiagonal."""
        B, d = x.size()
        if d % stride != 0:
            pad_size = stride - (d % stride)
            x = F.pad(x, (0, pad_size))
            d = x.size(1)
        
        return x.view(B, stride, d // stride).transpose(0, 1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        step: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of improved XAttention.
        
        Args:
            x: [B, L, D] input tensor
            mask: [B, L] attention mask (optional)
            return_attention: whether to return attention weights
            step: current step (for warmup)
            
        Returns:
            dict with output and optionally attention weights
        """
        B, L, D = x.shape
        
        # Compute adaptive parameters
        adaptive_params = self._compute_adaptive_parameters(x)
        
        # Handle warmup
        if self.use_adaptive_warmup:
            max_warmup = adaptive_params['warmup_steps'].mean().item()
        else:
            max_warmup = 5.0
        
        is_warmup = step < max_warmup
        
        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        outputs = []
        attention_weights = []
        
        for h in range(self.num_heads):
            # Get head-specific parameters
            if self.use_gradient_optimization:
                threshold = self.threshold_optimizer.forward(h)
            else:
                threshold = torch.sigmoid(self.thresholds[h]).item()
            
            if self.use_dynamic_blocks:
                block_size = self.block_sizer.get_block_size(h)
            else:
                block_size = self.default_block_size
            
            # Get target density
            if self.use_content_adaptive:
                target_density = adaptive_params['density'].mean().item()
            else:
                target_density = 0.1
            
            # Process each head
            q_h = q[:, h]
            k_h = k[:, h]
            v_h = v[:, h]
            
            batch_outputs = []
            batch_attention = []
            
            for b in range(B):
                q_b = q_h[b]
                k_b = k_h[b]
                v_b = v_h[b]
                
                # Handle padding
                pad_len = (block_size - L % block_size) % block_size
                if pad_len > 0:
                    q_b = F.pad(q_b, (0, 0, 0, pad_len))
                    k_b = F.pad(k_b, (0, 0, 0, pad_len))
                    v_b = F.pad(v_b, (0, 0, 0, pad_len))
                
                if is_warmup:
                    # Full attention during warmup
                    scale = 1.0 / np.sqrt(self.head_dim)
                    attn_scores = torch.matmul(q_b[:L], k_b[:L].transpose(-2, -1)) * scale
                    
                    if mask is not None:
                        attn_scores = attn_scores.masked_fill(~mask[b].unsqueeze(0), float('-inf'))
                    
                    attn_weights = F.softmax(attn_scores, dim=-1)
                    out_b = torch.matmul(attn_weights, v_b[:L])
                    
                else:
                    # Sparse attention
                    attn_approx = self._approximate_attention_map(q_b, k_b, block_size)
                    mask_blocks = self._select_blocks_adaptive(
                        attn_approx, threshold, target_density, block_size
                    )
                    mask_blocks = mask_blocks[:L, :L]
                    
                    scale = 1.0 / np.sqrt(self.head_dim)
                    attn_scores = torch.matmul(q_b[:L], k_b[:L].transpose(-2, -1)) * scale
                    attn_scores = attn_scores.masked_fill(~mask_blocks, float('-inf'))
                    
                    if mask is not None:
                        attn_scores = attn_scores.masked_fill(~mask[b].unsqueeze(0), float('-inf'))
                    
                    attn_weights = F.softmax(attn_scores, dim=-1)
                    out_b = torch.matmul(attn_weights, v_b[:L])
                
                batch_outputs.append(out_b)
                batch_attention.append(attn_weights[:L, :L])
            
            outputs.append(torch.stack(batch_outputs).transpose(0, 1))
            attention_weights.append(torch.stack(batch_attention))
        
        # Concatenate heads and project
        out = torch.cat(outputs, dim=-1)
        out = self.out_proj(out)
        
        # Update warmup counter
        if not is_warmup:
            self.warmup_steps[0] = max_warmup
        
        result = {'output': out}
        if return_attention:
            result['attention_weights'] = torch.stack(attention_weights, dim=1)
        
        return result
    
    def load_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Load pre-trained weights."""
        self.load_state_dict(state_dict)
    
    def get_sparsity_stats(self) -> Dict[str, Any]:
        """Get comprehensive sparsity statistics."""
        stats = {
            'default_block_size': self.default_block_size,
            'strides': self.strides,
            'use_adaptive_warmup': self.use_adaptive_warmup,
            'use_multi_scale': self.use_multi_scale,
            'use_content_adaptive': self.use_content_adaptive,
            'use_gradient_optimization': self.use_gradient_optimization,
            'use_dynamic_blocks': self.use_dynamic_blocks,
            'warmup_steps': self.warmup_steps.item()
        }
        
        if self.use_gradient_optimization:
            stats['thresholds'] = [self.threshold_optimizer.forward(h) for h in range(self.num_heads)]
        else:
            stats['thresholds'] = [torch.sigmoid(self.thresholds[h]).item() for h in range(self.num_heads)]
        
        return stats
    
    def optimize_thresholds(self, validation_loader, criterion, device='cuda'):
        """Optimize thresholds using gradient-based approach."""
        if not self.use_gradient_optimization:
            return
        
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        for batch_idx, (x, y) in enumerate(validation_loader):
            if batch_idx >= 100:  # Limit optimization steps
                break
                
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            outputs = self.forward(x)
            loss = criterion(outputs['output'], y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update thresholds based on gradients
            if hasattr(self.threshold_optimizer, 'thresholds'):
                gradients = self.threshold_optimizer.thresholds.grad
                if gradients is not None:
                    self.threshold_optimizer.update_thresholds(gradients)
            
            optimizer.step()
        
        self.eval()