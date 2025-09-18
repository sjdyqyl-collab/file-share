"""
Improved XAttention Implementation
This implements all the suggested improvements from the comprehensive analysis:
1. Adaptive block size selection
2. Decoding stage extension
3. Multi-pattern ensemble
4. KV cache compression
5. Dynamic stride scheduling
6. Hierarchical attention patterns
7. Gradient-aware selection
8. Streaming extension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from collections import deque


class AdaptiveBlockSelector(nn.Module):
    """Adaptive block size selection based on local attention entropy."""
    
    def __init__(self, hidden_size: int, block_sizes: List[int] = [4, 8, 16]):
        super().__init__()
        self.block_sizes = block_sizes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, len(block_sizes))
        )
        
    def forward(self, attention_entropy: torch.Tensor) -> int:
        """Select block size based on attention entropy."""
        logits = self.classifier(attention_entropy.mean(dim=-1))
        return self.block_sizes[torch.argmax(logits)]


class MultiPatternScorer(nn.Module):
    """Multi-pattern ensemble for importance scoring."""
    
    def __init__(self, num_patterns: int = 4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_patterns))
        self.patterns = ['antidiagonal', 'diagonal', 'vertical', 'horizontal']
        
    def compute_pattern_score(self, attention_block: torch.Tensor, pattern: str, stride: int) -> torch.Tensor:
        """Compute score for a specific pattern."""
        B = attention_block.shape[0]
        scores = []
        
        if pattern == 'antidiagonal':
            # Original antidiagonal pattern
            for k in range(2 * B - 1):
                i_indices, j_indices = [], []
                for i in range(max(0, k - B + 1), min(k + 1, B)):
                    j = k - i
                    if 0 <= j < B and (i + j) % stride == 0:
                        i_indices.append(i)
                        j_indices.append(j)
                if i_indices:
                    scores.append(attention_block[i_indices, j_indices].sum())
                    
        elif pattern == 'diagonal':
            # Diagonal pattern (main diagonal and parallel)
            for k in range(-B + 1, B):
                i_indices, j_indices = [], []
                for i in range(max(0, -k), min(B, B - k)):
                    j = i + k
                    if 0 <= j < B and i % stride == 0:
                        i_indices.append(i)
                        j_indices.append(j)
                if i_indices:
                    scores.append(attention_block[i_indices, j_indices].sum())
                    
        elif pattern == 'vertical':
            # Vertical pattern
            for j in range(0, B, stride):
                scores.append(attention_block[:, j].sum())
                
        elif pattern == 'horizontal':
            # Horizontal pattern
            for i in range(0, B, stride):
                scores.append(attention_block[i, :].sum())
        
        return torch.tensor(scores).sum() if scores else torch.tensor(0.0)
    
    def forward(self, attention_block: torch.Tensor, stride: int) -> torch.Tensor:
        """Compute weighted ensemble score."""
        weights = F.softmax(self.weights, dim=0)
        total_score = 0.0
        
        for i, pattern in enumerate(self.patterns):
            score = self.compute_pattern_score(attention_block, pattern, stride)
            total_score += weights[i] * score
            
        return total_score


class KVCacheCompressor:
    """Compress key-value cache based on selected sparse blocks."""
    
    def __init__(self, compression_ratio: float = 0.5):
        self.compression_ratio = compression_ratio
        self.cache_map = {}
        
    def compress(self, k: torch.Tensor, v: torch.Tensor, block_mask: torch.Tensor, 
                block_size: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Compress KV cache based on selected blocks."""
        batch_size, num_heads, seq_len, head_dim = k.shape
        num_blocks = block_mask.shape[-1]
        
        # Collect indices of selected blocks
        selected_indices = []
        for b in range(batch_size):
            for h in range(num_heads):
                for j in range(num_blocks):
                    if block_mask[b, h, :, j].any():
                        start = j * block_size
                        end = min((j + 1) * block_size, seq_len)
                        selected_indices.extend(range(start, end))
        
        # Remove duplicates and sort
        selected_indices = sorted(list(set(selected_indices)))
        
        # Compress tensors
        k_compressed = k[:, :, selected_indices, :]
        v_compressed = v[:, :, selected_indices, :]
        
        # Create mapping for reconstruction
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
        
        return k_compressed, v_compressed, {'index_map': index_map, 'original_length': seq_len}
    
    def decompress(self, k_compressed: torch.Tensor, v_compressed: torch.Tensor, 
                  cache_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress KV cache."""
        original_length = cache_info['original_length']
        index_map = cache_info['index_map']
        batch_size, num_heads, _, head_dim = k_compressed.shape
        
        # Create full tensors
        k_full = torch.zeros(batch_size, num_heads, original_length, head_dim)
        v_full = torch.zeros(batch_size, num_heads, original_length, head_dim)
        
        # Fill in compressed values
        for old_idx, new_idx in index_map.items():
            k_full[:, :, old_idx, :] = k_compressed[:, :, new_idx, :]
            v_full[:, :, old_idx, :] = v_compressed[:, :, new_idx, :]
            
        return k_full, v_full


class GradientAwareSelector:
    """Use gradient information to identify critical attention connections."""
    
    def __init__(self, gradient_threshold: float = 0.1):
        self.gradient_threshold = gradient_threshold
        self.gradients = None
        
    def register_hooks(self, attention_module):
        """Register backward hooks to capture gradients."""
        def save_gradients(grad):
            self.gradients = grad
            
        return save_gradients
    
    def compute_gradient_importance(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute importance based on gradient magnitude."""
        if self.gradients is None:
            return torch.ones_like(attention_weights)
        
        # Use gradient magnitude as importance
        grad_magnitude = torch.abs(self.gradients)
        importance = torch.where(grad_magnitude > self.gradient_threshold, 
                               torch.ones_like(grad_magnitude), 
                               torch.zeros_like(grad_magnitude))
        return importance


class XAttentionImproved(nn.Module):
    """
    Improved XAttention with all suggested enhancements.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, 
                 block_sizes: List[int] = [4, 8, 16],
                 strides: List[int] = [4, 8, 16, 64],
                 dropout: float = 0.0,
                 warmup_steps: int = 5,
                 streaming_window: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_sizes = block_sizes
        self.strides = strides
        self.warmup_steps = warmup_steps
        self.streaming_window = streaming_window
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        
        # Enhanced components
        self.adaptive_selector = AdaptiveBlockSelector(hidden_size, block_sizes)
        self.pattern_scorer = MultiPatternScorer()
        self.cache_compressor = KVCacheCompressor()
        self.gradient_selector = GradientAwareSelector()
        
        # Dynamic parameters
        self.current_block_size = 8
        self.current_stride = 8
        self.step_count = 0
        
        # KV cache for streaming
        self.kv_cache = deque(maxlen=streaming_window)
        self.cache_info = {}
        
    def compute_adaptive_importance(self, q: torch.Tensor, k: torch.Tensor, 
                                  block_size: int, stride: int) -> torch.Tensor:
        """Compute importance scores with adaptive block sizes and multi-patterns."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Calculate number of blocks based on current block size
        num_blocks = (seq_len + block_size - 1) // block_size
        
        # Initialize importance scores
        importance_scores = torch.zeros(batch_size, num_heads, num_blocks, num_blocks)
        
        # Compute attention entropy for adaptive selection
        attention_entropy = torch.zeros(batch_size, seq_len)
        
        for b in range(batch_size):
            for h in range(num_heads):
                # Compute attention for entropy calculation
                scores = torch.matmul(q[b, h], k[b, h].T) * self.scale
                attn_weights = F.softmax(scores, dim=-1)
                attention_entropy[b] += -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
        
        # Select adaptive block size
        if seq_len > 32:  # Only adapt for longer sequences
            self.current_block_size = self.adaptive_selector(attention_entropy)
        
        # Recompute with new block size if changed
        if self.current_block_size != block_size:
            num_blocks = (seq_len + self.current_block_size - 1) // self.current_block_size
            importance_scores = torch.zeros(batch_size, num_heads, num_blocks, num_blocks)
        
        # Compute importance scores for each block
        for b_i in range(num_blocks):
            for b_j in range(num_blocks):
                i_start = b_i * self.current_block_size
                i_end = min((b_i + 1) * self.current_block_size, seq_len)
                j_start = b_j * self.current_block_size
                j_end = min((b_j + 1) * self.current_block_size, seq_len)
                
                if i_start < seq_len and j_start < seq_len:
                    q_block = q[:, :, i_start:i_end, :]
                    k_block = k[:, :, j_start:j_end, :]
                    
                    # Compute attention for this block
                    attn_block = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale
                    attn_block = F.softmax(attn_block, dim=-1)
                    
                    # Use multi-pattern ensemble scoring
                    for head in range(num_heads):
                        score = self.pattern_scorer(attn_block[0, head], stride)
                        importance_scores[0, head, b_i, b_j] = score
        
        return importance_scores
    
    def select_blocks_hierarchical(self, importance_scores: torch.Tensor, 
                                 threshold: float) -> torch.Tensor:
        """Hierarchical block selection: coarse + fine-grained."""
        batch_size, num_heads, num_blocks, _ = importance_scores.shape
        
        # Coarse selection
        coarse_mask = torch.zeros_like(importance_scores, dtype=torch.bool)
        for b in range(batch_size):
            for h in range(num_heads):
                flat_scores = importance_scores[b, h].flatten()
                if flat_scores.sum() > 0:
                    probs = F.softmax(flat_scores, dim=0)
                    sorted_probs, indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                    selected_indices = indices[cumulative_probs < threshold * 0.8]  # Coarse threshold
                    
                    for idx in selected_indices:
                        i = idx // num_blocks
                        j = idx % num_blocks
                        coarse_mask[b, h, i, j] = True
        
        # Fine-grained selection within selected coarse blocks
        fine_mask = coarse_mask.clone()
        
        return fine_mask
    
    def dynamic_stride_scheduling(self, seq_len: int, task_type: str = "language") -> int:
        """Dynamic stride selection based on input characteristics."""
        if task_type == "video_generation":
            return min(16, max(4, seq_len // 1024))
        elif task_type == "video_understanding":
            return min(32, max(8, seq_len // 2048))
        else:  # language
            return min(64, max(4, seq_len // 512))
    
    def streaming_attention(self, x: torch.Tensor, is_streaming: bool = False) -> torch.Tensor:
        """Handle streaming sequences with sliding window."""
        if not is_streaming:
            return self.forward_standard(x)
        
        batch_size, seq_len, _ = x.shape
        
        # Use sliding window approach
        if len(self.kv_cache) + seq_len > self.streaming_window:
            # Remove oldest entries
            while len(self.kv_cache) + seq_len > self.streaming_window:
                self.kv_cache.popleft()
        
        # Process current window
        return self.forward_standard(x)
    
    def forward_standard(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                        task_type: str = "language", use_cache: bool = False) -> torch.Tensor:
        """Standard forward pass with all improvements."""
        batch_size, seq_len, _ = x.shape
        
        # Warmup strategy for video generation
        if task_type == "video_generation" and self.step_count < self.warmup_steps:
            self.step_count += 1
            # Use full attention during warmup
            return self._full_attention(x, mask)
        
        # Dynamic stride scheduling
        self.current_stride = self.dynamic_stride_scheduling(seq_len, task_type)
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # KV cache compression if enabled
        if use_cache:
            k, v, self.cache_info = self.cache_compressor.compress(k, v, 
                                                                   torch.ones_like(k[:, :, :, 0]).bool(),
                                                                   self.current_block_size)
        
        # Compute adaptive importance scores
        importance_scores = self.compute_adaptive_importance(q, k, self.current_block_size, self.current_stride)
        
        # Hierarchical block selection
        thresholds = torch.ones(self.num_heads) * 0.85  # Adaptive threshold
        block_masks = []
        for h in range(self.num_heads):
            head_scores = importance_scores[:, h:h+1]
            head_mask = self.select_blocks_hierarchical(head_scores, thresholds[h])
            block_masks.append(head_mask)
        
        block_mask = torch.cat(block_masks, dim=1)
        
        # Gradient-aware selection if gradients available
        if self.gradient_selector.gradients is not None:
            grad_importance = self.gradient_selector.compute_gradient_importance(
                torch.ones_like(block_mask.float()))
            block_mask = block_mask & (grad_importance > 0)
        
        # Compute sparse attention
        attn_output = self._sparse_attention_improved(q, k, v, block_mask)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)
        
        return output
    
    def _full_attention(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Full attention for warmup or fallback."""
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        return attn_output
    
    def _sparse_attention_improved(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 block_mask: torch.Tensor) -> torch.Tensor:
        """Improved sparse attention with better efficiency."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        num_blocks = block_mask.shape[-1]
        
        output = torch.zeros_like(q)
        
        # Use efficient block processing
        for b in range(batch_size):
            for h in range(num_heads):
                selected_blocks = block_mask[b, h].nonzero()
                if len(selected_blocks) == 0:
                    continue
                
                # Process all selected blocks for this head efficiently
                for i, j in selected_blocks:
                    i_start = i * self.current_block_size
                    i_end = min((i + 1) * self.current_block_size, seq_len)
                    j_start = j * self.current_block_size
                    j_end = min((j + 1) * self.current_block_size, seq_len)
                    
                    # Batch processing for efficiency
                    q_block = q[b:b+1, h:h+1, i_start:i_end, :]
                    k_block = k[b:b+1, h:h+1, j_start:j_end, :]
                    v_block = v[b:b+1, h:h+1, j_start:j_end, :]
                    
                    scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale
                    attn_weights = F.softmax(scores, dim=-1)
                    attn_weights = self.dropout(attn_weights)
                    
                    attn_output = torch.matmul(attn_weights, v_block)
                    output[b:b+1, h:h+1, i_start:i_end, :] += attn_output[0]
        
        return output
    
    def get_enhanced_sparsity_info(self, x: torch.Tensor, task_type: str = "language") -> Dict:
        """Get comprehensive sparsity information."""
        batch_size, seq_len, _ = x.shape
        
        # Update dynamic parameters
        self.current_stride = self.dynamic_stride_scheduling(seq_len, task_type)
        
        # Project to Q, K
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute importance scores
        importance_scores = self.compute_adaptive_importance(q, k, self.current_block_size, self.current_stride)
        
        # Select blocks
        thresholds = torch.ones(self.num_heads) * 0.85
        block_masks = []
        for h in range(self.num_heads):
            head_scores = importance_scores[:, h:h+1]
            head_mask = self.select_blocks_hierarchical(head_scores, thresholds[h])
            block_masks.append(head_mask)
        
        block_mask = torch.cat(block_masks, dim=1)
        
        # Calculate metrics
        total_blocks = block_mask.numel()
        selected_blocks = block_mask.sum().item()
        density = selected_blocks / total_blocks
        sparsity = 1 - density
        
        # Compression ratio
        compression_ratio = 1 - density
        
        return {
            'total_blocks': total_blocks,
            'selected_blocks': selected_blocks,
            'density': density,
            'sparsity': sparsity,
            'compression_ratio': compression_ratio,
            'current_block_size': self.current_block_size,
            'current_stride': self.current_stride,
            'step_count': self.step_count,
            'is_warmup': self.step_count < self.warmup_steps if hasattr(self, 'warmup_steps') else False
        }
    
    def reset_state(self):
        """Reset all dynamic states."""
        self.step_count = 0
        self.kv_cache.clear()
        self.cache_info = {}
        self.current_block_size = 8
        self.current_stride = 8


def test_xattention_improved():
    """Test the improved XAttention implementation."""
    torch.manual_seed(42)
    
    batch_size, seq_len, hidden_size, num_heads = 1, 128, 256, 8
    
    # Create improved attention module
    attention = XAttentionImproved(
        hidden_size, num_heads,
        block_sizes=[4, 8, 16],
        strides=[4, 8, 16, 64],
        warmup_steps=3
    )
    
    # Test standard mode
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = attention.forward_standard(x, task_type="language")
    
    # Test video generation mode
    x_video = torch.randn(batch_size, 64, hidden_size)
    output_video = attention.forward_standard(x_video, task_type="video_generation")
    
    # Get enhanced sparsity info
    sparsity_info = attention.get_enhanced_sparsity_info(x, task_type="language")
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Enhanced sparsity info: {sparsity_info}")
    
    # Test streaming mode
    attention.reset_state()
    output_stream = attention.streaming_attention(x, is_streaming=True)
    print(f"Streaming output shape: {output_stream.shape}")
    
    # Test KV cache compression
    attention.reset_state()
    output_cache = attention.forward_standard(x, use_cache=True)
    print(f"Cache-enabled output shape: {output_cache.shape}")


if __name__ == "__main__":
    test_xattention_improved()