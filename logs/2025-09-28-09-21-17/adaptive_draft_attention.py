import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
import numpy as np


class AdaptiveDraftAttention(nn.Module):
    """
    AdaptiveDraftAttention: Enhanced version with adaptive mechanisms
    
    Improvements over original DraftAttention:
    1. Content-aware sparsity patterns
    2. Progressive sparsity across denoising steps
    3. Multi-resolution draft attention
    4. Learnable pooling operators
    5. Head-specific sparsity patterns
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        sparsity_ratio: float = 0.9,
        pooling_kernels: List[Tuple[int, int]] = [(4, 8), (8, 16), (16, 32)],
        block_size: int = 128,
        use_full_attention_steps: float = 0.1,  # Reduced from 0.25
        adaptive_threshold: bool = True,
        head_specific_sparsity: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.base_sparsity_ratio = sparsity_ratio
        self.pooling_kernels = pooling_kernels
        self.block_size = block_size
        self.use_full_attention_steps = use_full_attention_steps
        self.adaptive_threshold = adaptive_threshold
        self.head_specific_sparsity = head_specific_sparsity
        
        # Validate dimensions
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Learnable pooling weights for adaptive pooling
        if self.adaptive_threshold:
            self.pooling_weights = nn.ParameterList([
                nn.Parameter(torch.ones(1, hidden_size, 1, 1) / (kh * kw))
                for kh, kw in pooling_kernels
            ])
        
        # Content analysis network for adaptive sparsity
        if self.adaptive_threshold:
            self.content_analyzer = nn.Sequential(
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Conv2d(hidden_size, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        # Head-specific sparsity parameters
        if self.head_specific_sparsity:
            self.head_sparsity = nn.Parameter(torch.ones(num_heads) * sparsity_ratio)
        
        # Scale factor for attention
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def _compute_motion_analysis(
        self,
        x: torch.Tensor,
        frame_size: Tuple[int, int],
        num_frames: int
    ) -> torch.Tensor:
        """Analyze temporal motion to guide adaptive sparsity."""
        B, L, D = x.shape
        H, W = frame_size
        
        # Reshape to video format
        video = x.view(B, num_frames, H, W, D).permute(0, 4, 1, 2, 3)  # (B, D, T, H, W)
        
        # Compute temporal differences
        if num_frames > 1:
            diff = torch.abs(video[:, :, 1:] - video[:, :, :-1])
            motion_score = torch.mean(diff, dim=[1, 2, 3, 4])  # (B,)
        else:
            motion_score = torch.zeros(B, device=x.device)
        
        return motion_score
    
    def _compute_content_complexity(
        self,
        x: torch.Tensor,
        frame_size: Tuple[int, int],
        num_frames: int
    ) -> torch.Tensor:
        """Analyze spatial complexity to guide adaptive sparsity."""
        B, L, D = x.shape
        H, W = frame_size
        
        # Reshape to spatial format
        spatial = x.view(B, num_frames, H, W, D).permute(0, 1, 4, 2, 3)  # (B, T, D, H, W)
        
        # Compute spatial gradients
        gradients = []
        for t in range(num_frames):
            frame = spatial[:, t]  # (B, D, H, W)
            
            # Sobel-like gradients
            grad_x = torch.abs(frame[:, :, :, 1:] - frame[:, :, :, :-1])
            grad_y = torch.abs(frame[:, :, 1:, :] - frame[:, :, :-1, :])
            
            # Average gradient magnitude
            grad_mag = torch.mean(grad_x) + torch.mean(grad_y)
            gradients.append(grad_mag)
        
        complexity = torch.stack(gradients).mean()
        return complexity
    
    def _get_progressive_sparsity(
        self,
        timestep_ratio: float,
        base_sparsity: float
    ) -> float:
        """Compute progressive sparsity based on timestep."""
        # Start with low sparsity, increase to target
        if timestep_ratio < 0.5:
            # Linear increase from 0 to base_sparsity
            sparsity = base_sparsity * (timestep_ratio * 2)
        else:
            # Constant after 50% of steps
            sparsity = base_sparsity
        
        return min(sparsity, base_sparsity)
    
    def _create_multi_resolution_draft(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        frame_size: Tuple[int, int],
        num_frames: int
    ) -> List[torch.Tensor]:
        """Create draft attention at multiple resolutions."""
        B, L, D = q.shape
        H, W = frame_size
        
        draft_attentions = []
        
        for i, (kh, kw) in enumerate(self.pooling_kernels):
            # Ensure divisibility
            if H % kh != 0 or W % kw != 0:
                continue
                
            # Reshape for pooling
            q_reshaped = q.view(B, num_frames, H, W, D)
            k_reshaped = k.view(B, num_frames, H, W, D)
            
            # Adaptive pooling with learnable weights
            if self.adaptive_threshold:
                weight = self.pooling_weights[i]
                q_pooled = F.conv2d(
                    q_reshaped.permute(0, 1, 4, 2, 3).reshape(-1, D, H, W),
                    weight.expand(D, -1, -1, -1),
                    stride=(kh, kw),
                    groups=D
                ).view(B, num_frames, D, H//kh, W//kw).permute(0, 1, 3, 4, 2)
                
                k_pooled = F.conv2d(
                    k_reshaped.permute(0, 1, 4, 2, 3).reshape(-1, D, H, W),
                    weight.expand(D, -1, -1, -1),
                    stride=(kh, kw),
                    groups=D
                ).view(B, num_frames, D, H//kh, W//kw).permute(0, 1, 3, 4, 2)
            else:
                # Standard average pooling
                q_pooled = F.avg_pool2d(
                    q_reshaped.permute(0, 1, 4, 2, 3).reshape(-1, D, H, W),
                    kernel_size=(kh, kw),
                    stride=(kh, kw)
                ).view(B, num_frames, D, H//kh, W//kw).permute(0, 1, 3, 4, 2)
                
                k_pooled = F.avg_pool2d(
                    k_reshaped.permute(0, 1, 4, 2, 3).reshape(-1, D, H, W),
                    kernel_size=(kh, kw),
                    stride=(kh, kw)
                ).view(B, num_frames, D, H//kh, W//kw).permute(0, 1, 3, 4, 2)
            
            # Flatten and compute attention
            g = num_frames * (H // kh) * (W // kw)
            q_draft = q_pooled.reshape(B, g, D)
            k_draft = k_pooled.reshape(B, g, D)
            
            draft_attention = torch.bmm(q_draft, k_draft.transpose(-2, -1)) * self.scale
            draft_attention = F.softmax(draft_attention, dim=-1)
            
            draft_attentions.append(draft_attention)
        
        return draft_attentions
    
    def _compute_reorder_indices(
        self, 
        frame_size: Tuple[int, int], 
        num_frames: int,
        pooling_kernel: Tuple[int, int],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate reorder and restore indices for patch-aligned processing."""
        H, W = frame_size
        h, w = pooling_kernel
        
        # Ensure dimensions are divisible
        if H % h != 0 or W % w != 0:
            # Use largest possible kernel
            h = min(h, H)
            w = min(w, W)
            while H % h != 0:
                h -= 1
            while W % w != 0:
                w -= 1
        
        # Generate reorder indices
        reorder_indices = []
        for f in range(num_frames):
            for i in range(H // h):
                for j in range(W // w):
                    for u in range(h):
                        for v in range(w):
                            y = i * h + u
                            x = j * w + v
                            idx = f * H * W + y * W + x
                            reorder_indices.append(idx)
        
        reorder_indices = torch.tensor(reorder_indices, dtype=torch.long, device=device)
        
        # Generate restore indices (inverse permutation)
        restore_indices = torch.empty_like(reorder_indices)
        restore_indices[reorder_indices] = torch.arange(len(reorder_indices), device=device)
        
        return reorder_indices, restore_indices
    
    def _create_adaptive_sparsity_mask(
        self,
        draft_attentions: List[torch.Tensor],
        frame_size: Tuple[int, int],
        num_frames: int,
        motion_score: torch.Tensor,
        complexity: torch.Tensor,
        timestep_ratio: float
    ) -> torch.Tensor:
        """Create content-adaptive sparsity mask."""
        B = draft_attentions[0].shape[0]
        H, W = frame_size
        n = num_frames * H * W
        
        # Select best resolution based on content
        resolution_idx = 1  # Default to medium resolution
        if complexity < 0.3:
            resolution_idx = 0  # Fine resolution for simple content
        elif complexity > 0.7:
            resolution_idx = 2  # Coarse resolution for complex content
        
        draft_attention = draft_attentions[resolution_idx]
        kh, kw = self.pooling_kernels[resolution_idx]
        
        # Adjust sparsity based on motion and timestep
        base_sparsity = self.base_sparsity_ratio
        if self.adaptive_threshold:
            # Reduce sparsity for high motion content
            motion_factor = 1.0 - (motion_score.mean() * 0.3)
            base_sparsity *= motion_factor
        
        # Progressive sparsity
        current_sparsity = self._get_progressive_sparsity(timestep_ratio, base_sparsity)
        
        # Create mask
        B, g, _ = draft_attention.shape
        flat_scores = draft_attention.view(B, -1)
        k = int(g * g * current_sparsity)
        
        # Get top-k indices
        _, top_indices = torch.topk(flat_scores, k=k, dim=-1)
        
        # Create region-level mask
        region_mask = torch.zeros_like(flat_scores)
        region_mask.scatter_(1, top_indices, 1.0)
        region_mask = region_mask.view(B, g, g)
        
        # Expand to token-level mask
        tokens_per_region = kh * kw
        token_mask = region_mask.repeat_interleave(tokens_per_region, dim=1)
        token_mask = token_mask.repeat_interleave(tokens_per_region, dim=2)
        
        # Ensure correct shape
        expected_shape = (B, n, n)
        if token_mask.shape != expected_shape:
            pad_n = expected_shape[1] - token_mask.shape[1]
            if pad_n > 0:
                token_mask = F.pad(token_mask, (0, pad_n, 0, pad_n))
        
        return token_mask
    
    def forward(
        self,
        x: torch.Tensor,
        frame_size: Tuple[int, int],
        num_frames: int,
        timestep_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Forward pass of AdaptiveDraftAttention.
        
        Args:
            x: Input tensor of shape (B, L, D)
            frame_size: (H, W) spatial dimensions per frame
            num_frames: Number of frames
            timestep_ratio: Current timestep ratio (0.0 to 1.0)
        
        Returns:
            Output tensor of shape (B, L, D)
        """
        B, L, D = x.shape
        H, W = frame_size
        
        # Validate input dimensions
        expected_L = num_frames * H * W
        assert L == expected_L, f"Expected sequence length {expected_L}, got {L}"
        
        # Default timestep ratio
        if timestep_ratio is None:
            timestep_ratio = 0.5
        
        # Use full attention for very initial steps
        if timestep_ratio < self.use_full_attention_steps:
            return self._full_attention(x)
        
        # Analyze content characteristics
        motion_score = self._compute_motion_analysis(x, frame_size, num_frames)
        complexity = self._compute_content_complexity(x, frame_size, num_frames)
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (B, L, D)
        k = self.k_proj(x)  # (B, L, D)
        v = self.v_proj(x)  # (B, L, D)
        
        # Use best pooling kernel for reordering
        best_kernel = self.pooling_kernels[1]  # Medium resolution
        reorder_indices, restore_indices = self._compute_reorder_indices(
            frame_size, num_frames, best_kernel, x.device
        )
        
        # Reorder tokens
        q_reordered = q[:, reorder_indices, :]
        k_reordered = k[:, reorder_indices, :]
        v_reordered = v[:, reorder_indices, :]
        
        # Multi-resolution draft attention
        draft_attentions = self._create_multi_resolution_draft(
            q_reordered, k_reordered, frame_size, num_frames
        )
        
        # Create adaptive sparsity mask
        sparsity_mask = self._create_adaptive_sparsity_mask(
            draft_attentions, frame_size, num_frames, motion_score, complexity, timestep_ratio
        )
        
        # Apply sparse attention
        output = self._sparse_attention(q_reordered, k_reordered, v_reordered, sparsity_mask)
        
        # Restore original order
        output_restored = torch.empty_like(output)
        output_restored[:, restore_indices, :] = output
        
        # Final projection
        return self.out_proj(output_restored)
    
    def _full_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Standard full attention computation."""
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        
        return self.out_proj(out)
    
    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply sparse attention with head-specific patterns."""
        B, L, D = q.shape
        
        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply sparsity mask
        if self.head_specific_sparsity and mask.shape[1] != self.num_heads:
            # Create head-specific masks
            head_masks = []
            for h in range(self.num_heads):
                head_sparsity = torch.sigmoid(self.head_sparsity[h]) * self.base_sparsity_ratio
                head_mask = self._adjust_mask_for_head(mask, head_sparsity)
                head_masks.append(head_mask)
            mask = torch.stack(head_masks, dim=1)
        else:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        
        return out
    
    def _adjust_mask_for_head(self, mask: torch.Tensor, sparsity: float) -> torch.Tensor:
        """Adjust mask for specific head sparsity."""
        B, L, _ = mask.shape
        
        # Calculate number of non-zero entries
        total_elements = L * L
        target_nonzero = int(total_elements * sparsity)
        
        # Get current non-zero indices
        flat_mask = mask.view(B, -1)
        current_nonzero = flat_mask.sum(dim=-1)
        
        # Adjust mask to match target sparsity
        adjusted_masks = []
        for b in range(B):
            mask_flat = flat_mask[b]
            nonzero_indices = mask_flat.nonzero().squeeze(-1)
            
            if len(nonzero_indices) > target_nonzero:
                # Keep top-k based on some heuristic
                keep_indices = nonzero_indices[:target_nonzero]
                new_mask = torch.zeros_like(mask_flat)
                new_mask[keep_indices] = 1.0
            else:
                new_mask = mask_flat
            
            adjusted_masks.append(new_mask.view(L, L))
        
        return torch.stack(adjusted_masks, dim=0)
    
    def load_weights(self, state_dict: dict):
        """Load pre-trained weights."""
        self.load_state_dict(state_dict)
    
    def get_sparsity_stats(self) -> dict:
        """Get adaptive sparsity statistics."""
        return {
            'base_sparsity_ratio': self.base_sparsity_ratio,
            'pooling_kernels': self.pooling_kernels,
            'block_size': self.block_size,
            'use_full_attention_steps': self.use_full_attention_steps,
            'adaptive_threshold': self.adaptive_threshold,
            'head_specific_sparsity': self.head_specific_sparsity,
            'learnable_params': sum(p.numel() for p in self.parameters())
        }