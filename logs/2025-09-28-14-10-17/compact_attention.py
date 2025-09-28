"""
Compact Attention: Hardware-aware acceleration for video diffusion transformers
Implements the core method from "Compact Attention: Exploiting Structured Spatio-Temporal Sparsity for Fast Video Generation"
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
        max_frames: int = 128,
        max_height: int = 64,
        max_width: int = 64,
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
        
        # Maximum dimensions for pre-computation
        self.max_frames = max_frames
        self.max_height = max_height
        self.max_width = max_width
        
        # Linear projections
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # Pre-computed masks cache
        self.register_buffer('mask_cache', torch.zeros(
            max_frames, num_heads, max_height, max_width, max_height, max_width
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
                    self.max_height, self.max_width, radius=3
                )
                self.mask_cache[t, h] = mask
                
        # Initialize temporal masks
        for g in range(self.temporal_groups):
            mask = self._create_temporal_mask(g, self.max_frames)
            self.temporal_masks[g] = mask
    
    def _create_local_pattern(self, height: int, width: int, radius: int) -> torch.Tensor:
        """Create local spherical neighborhood pattern."""
        mask = torch.zeros(height, width, height, width)
        
        for i in range(height):
            for j in range(width):
                # Local spherical neighborhood
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            if max(abs(di)/radius, abs(dj)/radius) <= 1:
                                mask[i, j, ni, nj] = 1.0
        return mask
    
    def _create_cross_pattern(self, height: int, width: int, 
                            h_width: int, v_width: int) -> torch.Tensor:
        """Create cross-shaped pattern with horizontal and vertical corridors."""
        mask = torch.zeros(height, width, height, width)
        
        for i in range(height):
            for j in range(width):
                # Horizontal corridor
                for dj in range(-h_width, h_width + 1):
                    nj = j + dj
                    if 0 <= nj < width:
                        mask[i, j, i, nj] = 1.0
                
                # Vertical corridor
                for di in range(-v_width, v_width + 1):
                    ni = i + di
                    if 0 <= ni < height:
                        mask[i, j, ni, j] = 1.0
        
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
                    mask[i, j] = 1.0 if 2 < distance <= 8 else 0.05
                elif group_id == 2:  # Far frames
                    mask[i, j] = 1.0 if 8 < distance <= 16 else 0.01
                else:  # Global
                    mask[i, j] = 0.5 if distance > 16 else 1.0
        
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
        
        # Reshape to tiles
        num_tiles = L_padded // tile_size
        q_tiles = q.view(B, H, num_tiles, tile_size, D)
        k_tiles = k.view(B, H, num_tiles, tile_size, D)
        v_tiles = v.view(B, H, num_tiles, tile_size, D)
        mask_tiles = mask.view(H, num_tiles, tile_size, num_tiles, tile_size)
        
        # Tile-based attention computation
        out_tiles = torch.zeros_like(q_tiles)
        
        for i in range(num_tiles):
            for j in range(num_tiles):
                if mask_tiles[:, i, :, j, :].sum() > 0:  # Non-zero mask
                    q_tile = q_tiles[:, :, i]  # [B, H, tile_size, D]
                    k_tile = k_tiles[:, :, j]  # [B, H, tile_size, D]
                    v_tile = v_tiles[:, :, j]  # [B, H, tile_size, D]
                    mask_tile = mask_tiles[:, i, :, j, :]  # [H, tile_size, tile_size]
                    
                    # Compute attention for this tile pair
                    scores = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * self.scale
                    scores = scores + mask_tile.unsqueeze(0).log()
                    attn = F.softmax(scores, dim=-1)
                    
                    out_tiles[:, :, i] += torch.matmul(attn, v_tile)
        
        # Reshape back to original sequence length
        out = out_tiles.view(B, H, L_padded, D)
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
        
        # Get spatial mask
        if frame_idx is None:
            frame_idx = 0
        spatial_mask = self.mask_cache[frame_idx % self.max_frames]  # [H, H, W, H, W]
        
        # Reshape spatial mask to 2D
        h = w = int(math.sqrt(L))
        if h * w == L:  # 2D spatial data
            spatial_mask = spatial_mask[:h, :w, :h, :w].reshape(L, L)
        else:  # 1D sequence
            spatial_mask = torch.eye(L, device=x.device)
        
        # Get temporal mask
        if temporal_group is None:
            temporal_group = 0
        temporal_mask = self.temporal_masks[temporal_group % self.temporal_groups]
        
        # Combine masks
        combined_mask = spatial_mask * temporal_mask[:L, :L]
        
        # Apply tile-based sparse attention
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
        max_frames: int = 128,
        max_height: int = 64,
        max_width: int = 64,
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
        
        # Cache for adaptive masks
        self.register_buffer('adaptive_masks', torch.zeros(
            max_frames, num_heads, max_height * max_width, max_height * max_width
        ))
        
    def _compute_content_complexity(self, x: torch.Tensor) -> torch.Tensor:
        """Compute content complexity via attention map variance."""
        B, L, D = x.shape
        
        # Compute attention statistics
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k = qkv[:, :, 0], qkv[:, :, 1]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Measure complexity via variance
        variance = torch.var(scores, dim=-1).mean(dim=(1, 2))
        return variance
    
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
        
        # Compute full attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Initialize masks
        masks = torch.ones(B, H, L, L, device=q.device)
        
        for b in range(B):
            # Sort attention scores for each head
            sorted_scores, indices = torch.sort(scores[b], dim=-1, descending=True)
            
            # Compute cumulative attention mass
            cumulative_mass = torch.cumsum(F.softmax(sorted_scores, dim=-1), dim=-1)
            
            # Apply adaptive thresholds
            recall_t = recall_threshold[b]
            cost_t = cost_threshold[b]
            
            for h in range(H):
                # Find threshold index for recall
                recall_indices = (cumulative_mass[h] >= recall_t).nonzero()
                if len(recall_indices) > 0:
                    min_recall_idx = recall_indices[0].item()
                    max_cost_idx = int(cost_t * L)
                    
                    # Use the more restrictive threshold
                    threshold_idx = min(min_recall_idx, max_cost_idx)
                    
                    # Create sparse mask
                    mask_indices = indices[h, :threshold_idx + 1]
                    masks[b, h, :, :] = 0.0
                    masks[b, h, :, mask_indices] = 1.0
        
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


# Utility functions for mask pre-computation
class CompactMaskSearcher:
    """
    Offline mask search algorithm for Compact Attention.
    Implements boundary contraction with dual threshold system.
    """
    
    def __init__(
        self,
        recall_threshold: float = 0.9,
        cost_threshold: float = 0.04,
        max_iterations: int = 100
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
        
        for iteration in range(self.max_iterations):
            # Compute current recall and cost
            current_recall = self._compute_recall(attention_maps, mask)
            current_cost = mask.sum().item() / (L * L)
            
            # Check termination criteria
            if current_recall < self.recall_threshold or current_cost > self.cost_threshold:
                break
            
            # Contract boundaries
            mask = self._contract_boundaries(mask, attention_maps, iteration)
        
        return mask
    
    def _compute_recall(self, attention_maps: torch.Tensor, mask: torch.Tensor) -> float:
        """Compute recall of preserved attention mass."""
        preserved_mass = (attention_maps * mask.unsqueeze(0)).sum()
        total_mass = attention_maps.sum()
        return (preserved_mass / total_mass).item()
    
    def _contract_boundaries(
        self, 
        mask: torch.Tensor, 
        attention_maps: torch.Tensor,
        iteration: int
    ) -> torch.Tensor:
        """Contract mask boundaries based on attention importance."""
        L = mask.shape[0]
        
        # Compute attention importance scores
        importance = attention_maps.mean(dim=0)  # [L, L]
        
        # Find least important connections
        threshold = torch.quantile(importance[mask > 0], 0.1)
        
        # Remove low-importance connections
        new_mask = mask.clone()
        new_mask[importance < threshold] = 0.0
        
        return new_mask


# Example usage and testing
if __name__ == "__main__":
    # Test Compact Attention
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    dim = 512
    seq_len = 1024
    batch_size = 2
    
    compact_attn = CompactAttention(dim=dim, num_heads=8).to(device)
    adaptive_attn = CompactAttentionWithAdaptiveThresholding(dim=dim, num_heads=8).to(device)
    
    # Create test input
    x = torch.randn(batch_size, seq_len, dim).to(device)
    
    # Test forward pass
    print("Testing Compact Attention...")
    with torch.no_grad():
        output1 = compact_attn(x, frame_idx=0, temporal_group=0)
        print(f"Compact Attention output shape: {output1.shape}")
        
        output2 = adaptive_attn(x, noise_level=0.5)
        print(f"Adaptive Compact Attention output shape: {output2.shape}")
    
    # Test mask searcher
    print("\nTesting Mask Searcher...")
    searcher = CompactMaskSearcher()
    dummy_attention = torch.softmax(torch.randn(8, seq_len, seq_len), dim=-1)
    mask = searcher.search_optimal_mask(dummy_attention)
    print(f"Optimized mask sparsity: {(mask == 0).float().mean().item():.2%}")
    
    print("All tests completed successfully!")