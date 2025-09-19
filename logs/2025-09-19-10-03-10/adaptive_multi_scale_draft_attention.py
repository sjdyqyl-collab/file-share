"""
Adaptive Multi-Scale Draft Attention (AMDA)
Improved version of DraftAttention with multi-scale pooling and adaptive scale selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class AdaptiveMultiScaleDraftAttention(nn.Module):
    """
    Adaptive Multi-Scale Draft Attention (AMDA) implementation.
    
    This improved version uses multiple pooling scales and dynamically selects
    the optimal scale for each attention module using a lightweight gating network.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        sparsity_ratio: float = 0.9,
        pooling_scales: List[Tuple[int, int]] = [(4, 8), (8, 16), (16, 32)],
        use_full_attention_steps: int = 0,
        **kwargs
    ):
        """
        Initialize AMDA module.
        
        Args:
            dim: Hidden dimension size
            num_heads: Number of attention heads
            sparsity_ratio: Target sparsity ratio
            pooling_scales: List of (spatial, temporal) pooling scales to use
            use_full_attention_steps: Number of initial steps to use full attention
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sparsity_ratio = sparsity_ratio
        self.pooling_scales = pooling_scales
        self.num_scales = len(pooling_scales)
        self.use_full_attention_steps = use_full_attention_steps
        
        # Ensure dimension is divisible by num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Gating network for scale selection
        self.gate_network = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, self.num_scales),
            nn.Softmax(dim=-1)
        )
        
        # Scale-specific draft networks
        self.draft_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Linear(dim // 2, dim // 4)
            ) for _ in range(self.num_scales)
        ])
        
        # Initialize weights
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
        
        for network in self.draft_networks:
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
    
    def _compute_multi_scale_draft_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        frame_size: Tuple[int, int],
        num_frames: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute draft attention at multiple scales.
        
        Args:
            q: Query tensor [batch, seq_len, dim]
            k: Key tensor [batch, seq_len, dim]
            frame_size: (height, width) of each frame
            num_frames: Number of frames
            
        Returns:
            combined_draft: Weighted combination of draft attentions
            gate_weights: Weights for each scale [batch, num_scales]
        """
        batch_size, seq_len, dim = q.shape
        height, width = frame_size
        
        # Get global statistics for gating
        q_mean = q.mean(dim=1)  # [batch, dim]
        k_mean = k.mean(dim=1)  # [batch, dim]
        gate_input = torch.cat([q_mean, k_mean], dim=-1)  # [batch, 2*dim]
        gate_weights = self.gate_network(gate_input)  # [batch, num_scales]
        
        all_draft_attentions = []
        
        for scale_idx, (patch_h, patch_w) in enumerate(self.pooling_scales):
            # Calculate number of patches for this scale
            num_patches_h = height // patch_h
            num_patches_w = width // patch_w
            g = num_frames * num_patches_h * num_patches_w
            
            # Reshape for pooling
            q_reshaped = q.view(batch_size, num_frames, height, width, dim)
            k_reshaped = k.view(batch_size, num_frames, height, width, dim)
            
            # Apply average pooling
            q_pooled = F.avg_pool3d(
                q_reshaped.permute(0, 4, 1, 2, 3),
                kernel_size=(1, patch_h, patch_w),
                stride=(1, patch_h, patch_w)
            ).permute(0, 2, 3, 4, 1).contiguous()
            
            k_pooled = F.avg_pool3d(
                k_reshaped.permute(0, 4, 1, 2, 3),
                kernel_size=(1, patch_h, patch_w),
                stride=(1, patch_h, patch_w)
            ).permute(0, 2, 3, 4, 1).contiguous()
            
            # Flatten and apply draft network
            q_draft = q_pooled.view(batch_size * g, dim)
            k_draft = k_pooled.view(batch_size * g, dim)
            
            q_draft = self.draft_networks[scale_idx](q_draft)
            k_draft = self.draft_networks[scale_idx](k_draft)
            
            q_draft = q_draft.view(batch_size, g, -1)
            k_draft = k_draft.view(batch_size, g, -1)
            
            # Compute draft attention
            scale_factor = 1.0 / np.sqrt(q_draft.shape[-1])
            attn_scores = torch.bmm(q_draft, k_draft.transpose(1, 2)) * scale_factor
            attn_draft = F.softmax(attn_scores, dim=-1)
            
            all_draft_attentions.append(attn_draft)
        
        # Weighted combination of draft attentions
        combined_draft = torch.zeros_like(all_draft_attentions[0])
        for i, draft_attn in enumerate(all_draft_attentions):
            weight = gate_weights[:, i:i+1, None]  # [batch, 1, 1]
            combined_draft += weight * draft_attn
        
        return combined_draft, gate_weights
    
    def _generate_adaptive_sparsity_mask(
        self,
        attn_draft: torch.Tensor,
        seq_len: int,
        frame_size: Tuple[int, int],
        num_frames: int,
        gate_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate adaptive sparsity mask based on combined draft attention.
        
        Args:
            attn_draft: Combined draft attention [batch, g, g]
            seq_len: Original sequence length
            frame_size: (height, width) of each frame
            num_frames: Number of frames
            gate_weights: Scale selection weights
            
        Returns:
            Binary sparsity mask [batch, seq_len, seq_len]
        """
        batch_size, g, _ = attn_draft.shape
        height, width = frame_size
        
        # Adaptive threshold based on gate weights
        base_keep_ratio = 1.0 - self.sparsity_ratio
        adaptive_keep_ratio = base_keep_ratio * (0.8 + 0.4 * gate_weights.max(dim=-1)[0].mean())
        
        num_keep = int(g * g * adaptive_keep_ratio)
        
        # Get top-k indices for each batch
        mask_draft = torch.zeros_like(attn_draft)
        
        for b in range(batch_size):
            flat_attn = attn_draft[b].flatten()
            _, top_indices = torch.topk(flat_attn, max(1, num_keep))
            
            row_indices = top_indices // g
            col_indices = top_indices % g
            
            mask_draft[b, row_indices, col_indices] = 1.0
        
        # Determine effective scale based on gate weights
        effective_scale_idx = torch.argmax(gate_weights, dim=-1)[0].item()
        patch_h, patch_w = self.pooling_scales[effective_scale_idx]
        
        # Expand mask to full resolution
        mask_full = torch.zeros(batch_size, seq_len, seq_len, device=attn_draft.device)
        
        # Calculate actual patch dimensions for this scale
        num_patches_h = height // patch_h
        num_patches_w = width // patch_w
        actual_g = num_frames * num_patches_h * num_patches_w
        
        # Resize mask if necessary (in case of different scales)
        if actual_g != g:
            # Interpolate mask to correct size
            mask_draft_resized = F.interpolate(
                mask_draft.unsqueeze(1),
                size=(actual_g, actual_g),
                mode='nearest'
            ).squeeze(1)
        else:
            mask_draft_resized = mask_draft
        
        # Apply mask to full resolution
        for b in range(batch_size):
            for i in range(actual_g):
                for j in range(actual_g):
                    if mask_draft_resized[b, i, j] > 0:
                        # Calculate token ranges
                        frame_i = i // (num_patches_h * num_patches_w)
                        patch_i = i % (num_patches_h * num_patches_w)
                        patch_row_i = patch_i // num_patches_w
                        patch_col_i = patch_i % num_patches_w
                        
                        frame_j = j // (num_patches_h * num_patches_w)
                        patch_j = j % (num_patches_h * num_patches_w)
                        patch_row_j = patch_j // num_patches_w
                        patch_col_j = patch_j % num_patches_w
                        
                        start_i = frame_i * height * width + patch_row_i * patch_h * width + patch_col_i * patch_w
                        end_i = min(start_i + patch_h * patch_w, seq_len)
                        
                        start_j = frame_j * height * width + patch_row_j * patch_h * width + patch_col_j * patch_w
                        end_j = min(start_j + patch_h * patch_w, seq_len)
                        
                        mask_full[b, start_i:end_i, start_j:end_j] = 1.0
        
        return mask_full
    
    def _progressive_reordering(
        self,
        x: torch.Tensor,
        frame_size: Tuple[int, int],
        num_frames: int,
        gate_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Progressive reordering based on selected scale.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            frame_size: (height, width) of each frame
            num_frames: Number of frames
            gate_weights: Scale selection weights
            
        Returns:
            reordered_x: Reordered tensor
            restore_indices: Indices to restore original order
        """
        batch_size, seq_len, dim = x.shape
        height, width = frame_size
        
        # Select optimal scale
        optimal_scale_idx = torch.argmax(gate_weights, dim=-1)[0].item()
        patch_h, patch_w = self.pooling_scales[optimal_scale_idx]
        
        # Create reordering indices based on optimal patch size
        indices = []
        for f in range(num_frames):
            for ph in range(0, height, patch_h):
                for pw in range(0, width, patch_w):
                    for h in range(ph, min(ph + patch_h, height)):
                        for w in range(pw, min(pw + patch_w, width)):
                            idx = f * height * width + h * width + w
                            indices.append(idx)
        
        indices = torch.tensor(indices, device=x.device, dtype=torch.long)
        
        # Ensure we have the right number of indices
        if len(indices) < seq_len:
            # Pad with remaining indices
            remaining = set(range(seq_len)) - set(indices.tolist())
            indices = torch.cat([indices, torch.tensor(list(remaining), device=x.device, dtype=torch.long)])
        
        reordered_x = x[:, indices, :]
        
        # Create restore indices
        restore_indices = torch.empty_like(indices)
        restore_indices[indices] = torch.arange(seq_len, device=x.device)
        
        return reordered_x, restore_indices
    
    def forward(
        self,
        x: torch.Tensor,
        frame_size: Tuple[int, int],
        num_frames: int,
        step_idx: int = 0,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of AMDA.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            frame_size: (height, width) of each frame
            num_frames: Number of frames
            step_idx: Current denoising step
            
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        batch_size, seq_len, dim = x.shape
        
        # Use full attention for initial steps
        if step_idx < self.use_full_attention_steps:
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            scale = 1.0 / np.sqrt(self.head_dim)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn_weights, v)
            
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
            return self.out_proj(out)
        
        # Multi-scale draft computation
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Compute multi-scale draft attention
        attn_draft, gate_weights = self._compute_multi_scale_draft_attention(q, k, frame_size, num_frames)
        
        # Generate adaptive sparsity mask
        sparsity_mask = self._generate_adaptive_sparsity_mask(attn_draft, seq_len, frame_size, num_frames, gate_weights)
        
        # Progressive reordering
        x_reordered, restore_indices = self._progressive_reordering(x, frame_size, num_frames, gate_weights)
        
        # Recompute projections for reordered tokens
        q_reordered = self.q_proj(x_reordered)
        k_reordered = self.k_proj(x_reordered)
        v_reordered = self.v_proj(x_reordered)
        
        # Multi-head attention with adaptive sparsity
        q_reordered = q_reordered.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_reordered = k_reordered.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_reordered = v_reordered.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = torch.matmul(q_reordered, k_reordered.transpose(-2, -1)) * scale
        
        # Apply sparsity mask
        sparsity_mask = sparsity_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn_scores = attn_scores.masked_fill(sparsity_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v_reordered)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        out = self.out_proj(out)
        
        # Restore original token order
        out_restored = out[:, restore_indices, :]
        
        return out_restored
    
    def load_weights(self, checkpoint_path: str):
        """Load pretrained weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint)
        print(f"Loaded weights from {checkpoint_path}")
    
    def save_weights(self, checkpoint_path: str):
        """Save current weights to checkpoint."""
        torch.save(self.state_dict(), checkpoint_path)
        print(f"Saved weights to {checkpoint_path}")


class ProgressiveSparsityScheduler(nn.Module):
    """
    Progressive Sparsity Scheduling (PSS) module.
    
    Varies sparsity ratio based on denoising timestep and content complexity.
    """
    
    def __init__(self, max_steps: int = 1000):
        super().__init__()
        self.max_steps = max_steps
        
        # Sparsity schedule parameters
        self.register_buffer('early_sparsity', torch.tensor(0.4))
        self.register_buffer('mid_sparsity', torch.tensor(0.8))
        self.register_buffer('late_sparsity', torch.tensor(0.7))
        
    def get_sparsity_ratio(self, step_idx: int, content_complexity: Optional[float] = None) -> float:
        """
        Get adaptive sparsity ratio for current step.
        
        Args:
            step_idx: Current denoising step
            content_complexity: Optional complexity metric [0, 1]
            
        Returns:
            Sparsity ratio for current step
        """
        progress = step_idx / self.max_steps
        
        if progress < 0.25:
            # Early steps: lower sparsity for structure establishment
            sparsity = self.early_sparsity.item()
        elif progress < 0.75:
            # Mid steps: higher sparsity for detail refinement
            sparsity = self.mid_sparsity.item()
        else:
            # Late steps: adaptive based on content
            sparsity = self.late_sparsity.item()
            if content_complexity is not None:
                sparsity = sparsity + 0.2 * (content_complexity - 0.5)
        
        return max(0.1, min(0.95, sparsity))


def test_adaptive_multi_scale_draft_attention():
    """Test function for AMDA."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test configuration
    batch_size = 2
    seq_len = 128 * 64  # 128 frames, 64x64 patches
    dim = 512
    num_heads = 8
    frame_size = (64, 64)
    num_frames = 128
    
    # Create model
    model = AdaptiveMultiScaleDraftAttention(
        dim=dim,
        num_heads=num_heads,
        sparsity_ratio=0.9,
        pooling_scales=[(4, 8), (8, 16), (16, 32)]
    ).to(device)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, dim).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x, frame_size=frame_size, num_frames=num_frames, step_idx=50)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test progressive sparsity scheduler
    scheduler = ProgressiveSparsityScheduler()
    for step in [0, 250, 500, 750, 999]:
        sparsity = scheduler.get_sparsity_ratio(step)
        print(f"Step {step}: sparsity = {sparsity:.3f}")
    
    return model, output


if __name__ == "__main__":
    test_adaptive_multi_scale_draft_attention()