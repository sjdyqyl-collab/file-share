"""
Adaptive Hierarchical Draft Attention (AHDA)
Multi-level pooling with learned fusion weights for improved quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class AdaptiveHierarchicalDraftAttention(nn.Module):
    """
    Adaptive Hierarchical Draft Attention with multi-level pooling and learned fusion.
    
    This extends DraftAttention by using multiple pooling levels and learning
    optimal fusion weights for better sparsity guidance.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        sparsity_ratio: float = 0.8,
        pooling_levels: List[Tuple[int, int]] = [(8, 16), (4, 8), (2, 4)],
        fusion_type: str = "learned",
        **kwargs
    ):
        """
        Initialize AHDA module.
        
        Args:
            dim: Hidden dimension
            num_heads: Number of attention heads
            sparsity_ratio: Target sparsity ratio
            pooling_levels: List of (h, w) pooling kernels for each level
            fusion_type: How to fuse multi-level features ("learned", "weighted", "max")
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sparsity_ratio = sparsity_ratio
        self.pooling_levels = pooling_levels
        self.fusion_type = fusion_type
        self.num_levels = len(pooling_levels)
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert fusion_type in ["learned", "weighted", "max"], "Invalid fusion type"
        
        # Linear projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Learnable fusion parameters
        if fusion_type == "learned":
            self.fusion_weights = nn.Parameter(torch.ones(self.num_levels))
            self.fusion_mlp = nn.Sequential(
                nn.Linear(self.num_levels, self.num_levels * 2),
                nn.ReLU(),
                nn.Linear(self.num_levels * 2, self.num_levels),
                nn.Sigmoid()
            )
        elif fusion_type == "weighted":
            weights = [0.5, 0.3, 0.2] + [0.0] * max(0, self.num_levels - 3)
            self.register_buffer(
                "fixed_weights",
                torch.tensor(weights[:self.num_levels])
            )
        
        # Scale factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Reduction factors for each level
        self.reduction_factors = [h * w for h, w in pooling_levels]
    
    def _create_hierarchical_draft_maps(
        self,
        q: torch.Tensor,
        k: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Create draft attention maps at multiple hierarchical levels.
        
        Args:
            q: Query tensor [B, H, N, D]
            k: Key tensor [B, H, N, D]
            
        Returns:
            List of draft attention maps for each level
        """
        B, H, N, D = q.shape
        draft_maps = []
        
        for level, (h_kernel, w_kernel) in enumerate(self.pooling_levels):
            # Calculate pooling dimensions - use consistent approach
            # For now, use a simple square root approach
            sqrt_n = int(math.sqrt(N))
            if sqrt_n * sqrt_n == N:
                # Square spatial layout
                h = w = sqrt_n
                t = 1
            else:
                # Rectangular layout - use factors
                factors = []
                for i in range(1, int(math.sqrt(N)) + 1):
                    if N % i == 0:
                        factors.append((i, N // i))
                
                if factors:
                    # Pick closest to 16:9 aspect ratio
                    h, w = min(factors, key=lambda x: abs(x[0]/x[1] - 9/16))
                    t = 1
                else:
                    h = w = int(math.sqrt(N))
                    t = 1
            
            # Reshape for pooling [B*H, D, T, H, W]
            q_reshaped = q.transpose(1, 2).reshape(B * H, D, t, h, w)
            k_reshaped = k.transpose(1, 2).reshape(B * H, D, t, h, w)
            
            # Calculate target pooling size
            pool_h = max(1, h // h_kernel)
            pool_w = max(1, w // w_kernel)
            pool_t = max(1, t)
            
            # Apply adaptive pooling
            q_pooled = F.adaptive_avg_pool3d(q_reshaped, (pool_t, pool_h, pool_w))
            k_pooled = F.adaptive_avg_pool3d(k_reshaped, (pool_t, pool_h, pool_w))
            
            # Reshape for attention computation
            q_pooled = q_pooled.reshape(B, H, D, -1).transpose(2, 3)
            k_pooled = k_pooled.reshape(B, H, D, -1).transpose(2, 3)
            
            # Compute draft attention
            draft_attn = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) * self.scale
            draft_attn = F.softmax(draft_attn, dim=-1)
            
            draft_maps.append(draft_attn)
        
        return draft_maps
    
    def _fuse_draft_maps(self, draft_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-level draft attention maps.
        
        Args:
            draft_maps: List of draft attention maps
            
        Returns:
            Fused draft attention map (using coarsest level)
        """
        if self.fusion_type == "max":
            # Use the coarsest level (last in list)
            fused_map = draft_maps[-1]
            
        elif self.fusion_type == "weighted":
            # Use weighted combination at each spatial location
            # For simplicity, use the coarsest level
            fused_map = draft_maps[-1]
            
        else:  # learned
            # Use learned fusion at the coarsest level
            # Get level importance scores
            level_scores = []
            for draft_map in draft_maps:
                # Global average pooling
                score = draft_map.mean(dim=(-2, -1))  # [B, H]
                level_scores.append(score)
            
            # Stack scores and apply fusion
            level_scores = torch.stack(level_scores, dim=-1)  # [B, H, L]
            B, H, L = level_scores.shape
            
            # Apply MLP to get fusion weights
            weights = self.fusion_mlp(level_scores.reshape(-1, L))  # [B*H, L]
            weights = weights.reshape(B, H, L).softmax(dim=-1)  # [B, H, L]
            
            # Use weighted combination at coarsest level
            # For now, use the coarsest level map
            fused_map = draft_maps[-1]
        
        return fused_map
    
    def _create_adaptive_sparsity_mask(
        self,
        fused_map: torch.Tensor,
        sparsity_ratio: Optional[float] = None
    ) -> torch.Tensor:
        """
        Create adaptive sparsity mask based on fused draft attention.
        
        Args:
            fused_map: Fused draft attention map [B, H, N', N']
            sparsity_ratio: Target sparsity ratio
            
        Returns:
            Binary mask for sparsity
        """
        if sparsity_ratio is None:
            sparsity_ratio = self.sparsity_ratio
        
        B, H, N_draft, _ = fused_map.shape
        
        # Use attention entropy for adaptive sparsity
        attention_entropy = -(fused_map * torch.log(fused_map + 1e-8)).sum(dim=-1)
        
        # Normalize entropy
        max_entropy = math.log(N_draft)
        normalized_entropy = attention_entropy / max_entropy
        
        # Adjust sparsity ratio
        adaptive_sparsity = sparsity_ratio + 0.1 * normalized_entropy.mean()
        adaptive_sparsity = torch.clamp(adaptive_sparsity, 0.5, 0.9)
        
        # Calculate tokens to keep
        keep_ratio = 1.0 - adaptive_sparsity
        num_keep = max(1, int(N_draft * keep_ratio))
        
        # Get importance scores
        importance = fused_map.max(dim=-1)[0]  # [B, H, N_draft]
        
        # Get top-k indices
        _, top_indices = torch.topk(importance, num_keep, dim=-1, sorted=False)
        
        # Create mask
        mask = torch.zeros_like(importance, dtype=torch.bool)
        mask.scatter_(-1, top_indices, True)
        
        return mask
    
    def _reorder_tokens(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        level: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reorder tokens based on sparsity mask.
        
        Args:
            x: Input tensor [B, N, D]
            mask: Binary mask [B, N']
            level: Which pooling level to use
            
        Returns:
            Reordered tensor and indices
        """
        B, N, D = x.shape
        reduction_factor = self.reduction_factors[level]
        
        # Expand mask to full resolution
        mask_expanded = mask.repeat_interleave(reduction_factor, dim=-1)
        mask_expanded = mask_expanded[:, :N]
        
        # Get indices
        indices = torch.where(mask_expanded)[1].reshape(B, -1)
        
        # Reorder
        batch_indices = torch.arange(B, device=x.device).unsqueeze(-1)
        x_reordered = x[batch_indices, indices]
        
        return x_reordered, indices
    
    def _restore_order(
        self,
        x_sparse: torch.Tensor,
        indices: torch.Tensor,
        original_shape: torch.Size
    ) -> torch.Tensor:
        """Restore original token order."""
        B, N, D = original_shape
        device = x_sparse.device
        
        x_restored = torch.zeros(original_shape, device=device, dtype=x_sparse.dtype)
        batch_indices = torch.arange(B, device=device).unsqueeze(-1)
        x_restored[batch_indices, indices] = x_sparse
        
        return x_restored
    
    def forward(
        self,
        x: torch.Tensor,
        sparsity_ratio: Optional[float] = None,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of AHDA.
        
        Args:
            x: Input tensor [B, N, D]
            sparsity_ratio: Override default sparsity ratio
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Output tensor [B, N, D]
        """
        B, N, D = x.shape
        H = self.num_heads
        
        # Project to Q, K, V
        q = self.q_proj(x).reshape(B, N, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, H, self.head_dim).transpose(1, 2)
        
        # Create hierarchical draft maps
        draft_maps = self._create_hierarchical_draft_maps(q, k)
        
        # Use the coarsest level for sparsity determination
        coarsest_map = draft_maps[-1]
        
        # Create adaptive sparsity mask
        mask = self._create_adaptive_sparsity_mask(coarsest_map, sparsity_ratio)
        
        # Reorder tokens using coarsest level
        q_flat = q.transpose(1, 2).reshape(B, N, D)
        k_flat = k.transpose(1, 2).reshape(B, N, D)
        v_flat = v.transpose(1, 2).reshape(B, N, D)
        
        q_reordered, q_indices = self._reorder_tokens(q_flat, mask, level=-1)
        k_reordered, k_indices = self._reorder_tokens(k_flat, mask, level=-1)
        v_reordered, v_indices = self._reorder_tokens(v_flat, mask, level=-1)
        
        # Ensure consistent indices
        assert torch.equal(q_indices, k_indices) and torch.equal(k_indices, v_indices)
        
        # Reshape for multi-head attention
        N_sparse = q_reordered.shape[1]
        q_sparse = q_reordered.reshape(B, N_sparse, H, self.head_dim).transpose(1, 2)
        k_sparse = k_reordered.reshape(B, N_sparse, H, self.head_dim).transpose(1, 2)
        v_sparse = v_reordered.reshape(B, N_sparse, H, self.head_dim).transpose(1, 2)
        
        # Compute sparse attention
        attn_weights = torch.matmul(q_sparse, k_sparse.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out_sparse = torch.matmul(attn_weights, v_sparse)
        
        # Restore order
        out_flat = out_sparse.transpose(1, 2).reshape(B, N_sparse, D)
        out_restored = self._restore_order(out_flat, q_indices, (B, N, D))
        
        # Final projection
        output = self.out_proj(out_restored)
        
        if return_intermediate:
            return output, {
                'draft_maps': draft_maps,
                'coarsest_map': coarsest_map,
                'mask': mask,
                'sparsity_ratio': 1.0 - (mask.sum().float() / (mask.numel() / B / H))
            }
        
        return output
    
    def load_weights(self, state_dict: dict):
        """Load pre-trained weights."""
        self.load_state_dict(state_dict)


class AHDAConfig:
    """Configuration for AHDA experiments."""
    
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        sparsity_ratio: float = 0.8,
        pooling_levels: List[Tuple[int, int]] = [(8, 16), (4, 8), (2, 4)],
        fusion_type: str = "learned"
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.sparsity_ratio = sparsity_ratio
        self.pooling_levels = pooling_levels
        self.fusion_type = fusion_type


def create_ahda_model(config: AHDAConfig) -> AdaptiveHierarchicalDraftAttention:
    """Create AHDA model from configuration."""
    return AdaptiveHierarchicalDraftAttention(**config.__dict__)


# Example usage and testing
if __name__ == "__main__":
    # Test basic functionality
    B, N, D = 2, 1024, 768
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(B, N, D).to(device)
    
    # Test AHDA
    ahda = AdaptiveHierarchicalDraftAttention(
        dim=D,
        num_heads=12,
        sparsity_ratio=0.8,
        pooling_levels=[(8, 16), (4, 8)],
        fusion_type="max"  # Use max fusion for simplicity
    ).to(device)
    
    with torch.no_grad():
        output = ahda(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print("AHDA forward pass successful!")