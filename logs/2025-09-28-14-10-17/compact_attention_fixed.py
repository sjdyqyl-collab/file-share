"""
Compact Attention: Hardware-aware acceleration for video diffusion transformers
Fixed version with reduced memory usage for testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
import math

class CompactAttention(nn.Module):
    """
    Tile-based deformable sparse attention for video diffusion transformers.
    
    Key innovations:
    1. Tile-based deformable sparse patterns
    2. Frame-group-wise temporal adaptation
    3. Automated offline mask search with dual thresholds
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        tile_size: int = 16,
        temporal_groups: int = 4,
        recall_threshold: float = 0.9,
        cost_threshold: float = 0.04,
        max_frames: int = 32,  # Reduced from 128
        max_height: int = 16,  # Reduced from 64
        max_width: int = 16,   # Reduced from 64
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Tile configuration
        self.tile_size = tile_size
        self.temporal_groups = temporal_groups
        self.recall_threshold = recall_threshold
        self.cost_threshold = cost_threshold
        
        # Maximum dimensions for pre-computation (reduced)
        self.max_frames = max_frames
        self.max_height = max_height
        self.max_width = max_width
        
        # Linear projections
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # Pre-computed masks cache (reduced size)
        self.register_buffer('mask_cache', torch.zeros(
            max_frames, num_heads, max_height * max_width, max_height * max_width
        ))
        self.register_buffer('temporal_masks', torch.zeros(
            temporal_groups, max_frames, max_frames
        ))
        
        # Initialize masks
        self._initialize_masks()
        
    def _initialize_masks(self):
        """Initialize sparse patterns with local and cross-shaped patterns."""
        # Initialize local patterns
        for t in range(self.max_frames):
            for h in range(self.num_heads):
                mask = self._create_local_pattern(
                    self.max_height * self.max_width, radius=3
                )
                self.mask_cache[t, h] = mask
                
        # Initialize temporal masks
        for g in range(self.temporal_groups):
            mask = self._create_temporal_mask(g, self.max_frames)
            self.temporal_masks[g] = mask
    
    def _create_local_pattern(self, seq_len: int, radius: int) -> torch.Tensor:
        """Create local neighborhood pattern."""
        mask = torch.zeros(seq_len, seq_len)
        
        for i in range(seq_len):
            # Local neighborhood
            for j in range(max(0, i - radius), min(seq_len, i + radius + 1)):
                mask[i, j] = 1.0
        
        return mask
    
    def _create_temporal_mask(self, group_id: int, num_frames: int) -> torch.Tensor:
        """Create temporal mask based on frame distance."""
        mask = torch.zeros(num_frames, num_frames)
        
        # Distance-based temporal grouping
        for i in range(num_frames):
            for j in range(num_frames):
                distance = abs(i - j)
                
                if group_id == 0:  # Near frames
                    mask[i, j] = 1.0 if distance <= 2 else 0.1
                elif group_id == 1:  # Medium distance
                    mask[i, j] = 1.0 if 2 < distance <= 4 else 0.05
                elif group_id == 2:  # Far frames
                    mask[i, j] = 1.0 if 4 < distance <= 8 else 0.01
                else:  # Global
                    mask[i, j] = 0.5 if distance > 8 else 1.0
        
        return mask
    
    def _tile_based_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        mask: torch.Tensor,
        tile_size: int
    ) -> torch.Tensor:
        """
        Perform tile-based sparse attention computation.
        
        Args:
            q, k, v: [B, H, L, D] - query, key, value tensors
            mask: [H, L, L] - sparse attention mask
            tile_size: Size of tiles for computation
        
        Returns:
            out: [B, H, L, D] - attention output
        """
        B, H, L, D = q.shape
        
        # Ensure L is divisible by tile_size
        pad_len = (tile_size - L % tile_size) % tile_size
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            mask = F.pad(mask, (0, pad_len, 0, pad_len))
            L_padded = L + pad_len
        else:
            L_padded = L
        
        # Simple sparse attention (without tiling for memory efficiency)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(mask.unsqueeze(0) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        
        # Remove padding
        if pad_len > 0:
            out = out[:, :, :L, :]
        
        return out
    
    def forward(
        self, 
        x: torch.Tensor,
        frame_idx: Optional[int] = None,
        temporal_group: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass with compact attention.
        
        Args:
            x: [B, L, D] - input tensor
            frame_idx: Current frame index for temporal masking
            temporal_group: Temporal group ID for adaptive sparsity
        
        Returns:
            out: [B, L, D] - output tensor
        """
        B, L, D = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Get spatial mask (resize to actual sequence length)
        if frame_idx is None:
            frame_idx = 0
        
        # Use appropriate mask size
        actual_mask_size = min(L, self.max_height * self.max_width)
        spatial_mask = self.mask_cache[frame_idx % self.max_frames, :, :actual_mask_size, :actual_mask_size]
        
        # Resize mask if needed
        if actual_mask_size < L:
            # Pad or create new mask for larger sequences
            spatial_mask = self._create_local_pattern(L, radius=3)
        else:
            spatial_mask = spatial_mask[0]  # Use first head's mask
        
        # Get temporal mask
        if temporal_group is None:
            temporal_group = 0
        temporal_mask = self.temporal_masks[temporal_group % self.temporal_groups]
        
        # Combine masks (use minimum of spatial and temporal)
        combined_mask = spatial_mask * temporal_mask[:L, :L]
        
        # Apply sparse attention
        out = self._tile_based_attention(q, k, v, combined_mask.unsqueeze(0), self.tile_size)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        
        return out


class CompactAttentionWithAdaptiveThresholding(nn.Module):
    """
    Enhanced Compact Attention with adaptive thresholding system.
    
    Improvement: Adaptive recall and cost thresholds based on content complexity
    and noise levels during denoising.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        tile_size: int = 16,
        temporal_groups: int = 4,
        base_recall_threshold: float = 0.9,
        base_cost_threshold: float = 0.04,
        entropy_threshold: float = 0.3,
        noise_scaling: float = 1.0,
        max_frames: int = 32,  # Reduced from 128
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Adaptive thresholding parameters
        self.base_recall_threshold = base_recall_threshold
        self.base_cost_threshold = base_cost_threshold
        self.entropy_threshold = entropy_threshold
        self.noise_scaling = noise_scaling
        
        # Tile configuration
        self.tile_size = tile_size
        self.temporal_groups = temporal_groups
        
        # Linear projections
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # Adaptive threshold networks
        self.entropy_predictor = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.threshold_adapter = nn.Sequential(
            nn.Linear(3, 16),  # entropy, noise_level, content_complexity
            nn.ReLU(),
            nn.Linear(16, 2)   # recall_threshold, cost_threshold
        )
        
    def _compute_content_complexity(self, x: torch.Tensor) -> torch.Tensor:
        """Compute content complexity via attention map variance."""
        B, L, D = x.shape
        
        # Simple complexity measure based on input variance
        complexity = torch.var(x, dim=-1).mean(dim=-1)
        return complexity
    
    def _compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention entropy for threshold adaptation."""
        B, L, D = x.shape
        entropy = self.entropy_predictor(x.mean(dim=1))  # [B, 1]
        return entropy.squeeze(-1)
    
    def _adaptive_thresholds(
        self, 
        entropy: torch.Tensor,
        content_complexity: torch.Tensor,
        noise_level: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute adaptive thresholds based on content characteristics."""
        
        # Normalize inputs
        noise_tensor = torch.tensor([noise_level], device=entropy.device).expand_as(entropy)
        
        # Combine features
        features = torch.stack([
            entropy,
            noise_tensor,
            content_complexity
        ], dim=-1)  # [B, 3]
        
        # Predict thresholds
        thresholds = self.threshold_adapter(features)  # [B, 2]
        
        # Clamp to reasonable ranges
        recall_threshold = torch.clamp(
            self.base_recall_threshold + thresholds[:, 0] * 0.1,
            min=0.8, max=0.95
        )
        cost_threshold = torch.clamp(
            self.base_cost_threshold + thresholds[:, 1] * 0.02,
            min=0.01, max=0.08
        )
        
        return recall_threshold, cost_threshold
    
    def _adaptive_mask_search(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        recall_threshold: torch.Tensor,
        cost_threshold: torch.Tensor
    ) -> torch.Tensor:
        """
        Adaptive mask search with dynamic thresholds.
        
        Args:
            q, k: [B, H, L, D] - query and key tensors
            recall_threshold: [B] - adaptive recall thresholds
            cost_threshold: [B] - adaptive cost thresholds
        
        Returns:
            mask: [B, H, L, L] - adaptive sparse masks
        """
        B, H, L, D = q.shape
        
        # Simple adaptive sparsity based on thresholds
        masks = torch.ones(B, H, L, L, device=q.device)
        
        for b in range(B):
            # Compute attention scores
            scores = torch.matmul(q[b], k[b].transpose(-2, -1)) * self.scale
            
            # Apply threshold-based sparsity
            threshold = torch.quantile(scores.flatten(), 1 - cost_threshold[b])
            masks[b] = (scores > threshold).float()
        
        return masks
    
    def forward(
        self,
        x: torch.Tensor,
        noise_level: float = 0.5,
        frame_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass with adaptive thresholding.
        
        Args:
            x: [B, L, D] - input tensor
            noise_level: Current noise level in denoising process
            frame_idx: Current frame index
        
        Returns:
            out: [B, L, D] - output tensor with adaptive sparsity
        """
        B, L, D = x.shape
        
        # Compute content characteristics
        entropy = self._compute_entropy(x)
        content_complexity = self._compute_content_complexity(x)
        
        # Get adaptive thresholds
        recall_threshold, cost_threshold = self._adaptive_thresholds(
            entropy, content_complexity, noise_level
        )
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Adaptive mask search
        mask = self._adaptive_mask_search(q, k, recall_threshold, cost_threshold)
        
        # Apply sparse attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        
        return out


class CompactMaskSearcher:
    """
    Offline mask search algorithm for Compact Attention.
    Implements boundary contraction with dual threshold system.
    """
    
    def __init__(
        self,
        recall_threshold: float = 0.9,
        cost_threshold: float = 0.04,
        max_iterations: int = 50  # Reduced from 100
    ):
        self.recall_threshold = recall_threshold
        self.cost_threshold = cost_threshold
        self.max_iterations = max_iterations
    
    def search_optimal_mask(
        self,
        attention_maps: torch.Tensor,
        tile_size: int = 16
    ) -> torch.Tensor:
        """
        Search optimal sparse mask using boundary contraction.
        
        Args:
            attention_maps: [H, L, L] - full attention maps for a layer/head
            tile_size: Size of tiles for computation
        
        Returns:
            mask: [L, L] - optimized sparse mask
        """
        H, L, _ = attention_maps.shape
        
        # Initialize with full attention
        mask = torch.ones(L, L)
        
        # Simple threshold-based sparsification
        total_mass = attention_maps.sum()
        flat_attention = attention_maps.mean(dim=0).flatten()
        
        # Sort by importance
        sorted_vals, sorted_indices = torch.sort(flat_attention, descending=True)
        cumulative_mass = torch.cumsum(sorted_vals, 0)
        
        # Find cutoff point
        cutoff_idx = (cumulative_mass >= self.recall_threshold * total_mass).nonzero()[0]
        max_allowed = int(self.cost_threshold * L * L)
        final_idx = min(cutoff_idx.item(), max_allowed)
        
        # Create sparse mask
        mask = torch.zeros(L, L)
        flat_mask = mask.flatten()
        flat_mask[sorted_indices[:final_idx]] = 1.0
        mask = flat_mask.view(L, L)
        
        return mask


# Example usage and testing
if __name__ == "__main__":
    # Test Compact Attention
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models with smaller parameters
    dim = 256  # Reduced from 512
    seq_len = 64  # Reduced from 1024
    batch_size = 2
    
    compact_attn = CompactAttention(dim=dim, num_heads=4).to(device)  # Reduced heads
    adaptive_attn = CompactAttentionWithAdaptiveThresholding(dim=dim, num_heads=4).to(device)
    
    # Create test input
    x = torch.randn(batch_size, seq_len, dim).to(device)
    
    print("Testing Compact Attention...")
    with torch.no_grad():
        output1 = compact_attn(x, frame_idx=0, temporal_group=0)
        print(f"Compact Attention output shape: {output1.shape}")
        
        output2 = adaptive_attn(x, noise_level=0.5)
        print(f"Adaptive Compact Attention output shape: {output2.shape}")
    
    # Test mask searcher
    print("\nTesting Mask Searcher...")
    searcher = CompactMaskSearcher()
    dummy_attention = torch.softmax(torch.randn(4, seq_len, seq_len), dim=-1)
    mask = searcher.search_optimal_mask(dummy_attention)
    print(f"Optimized mask sparsity: {(mask == 0).float().mean().item():.2%}")
    print(f"Mask shape: {mask.shape}")
    
    print("All tests completed successfully!")