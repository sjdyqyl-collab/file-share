import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class DraftAttention(nn.Module):
    """
    Original DraftAttention implementation from the paper.
    
    Implements two-stage attention:
    1. Low-resolution draft attention via average pooling
    2. Guided sparse attention on full-resolution sequence
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        sparsity_ratio: float = 0.9,
        kernel_size: Tuple[int, int] = (8, 16),
        max_sequence_length: int = 8192,
    ):
        """
        Args:
            hidden_size: Hidden dimension size (d)
            num_heads: Number of attention heads
            sparsity_ratio: Sparsity ratio r ∈ (0,1)
            kernel_size: Pooling kernel (h, w) for spatial pooling
            max_sequence_length: Maximum sequence length for buffer allocation
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.sparsity_ratio = sparsity_ratio
        self.kernel_size = kernel_size
        self.kernel_h, self.kernel_w = kernel_size
        
        # Ensure hidden_size is divisible by num_heads
        assert hidden_size % num_heads == 0, f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Scale factor for attention
        self.scale = math.sqrt(self.head_dim)
        
        # Buffers for reordering indices
        self.register_buffer('reorder_indices', None)
        self.register_buffer('restore_indices', None)
        
    def generate_reorder_indices(self, frame_size: Tuple[int, int], frames: int, device: torch.device):
        """Generate reorder indices for spatial locality."""
        H, W = frame_size
        h, w = self.kernel_h, self.kernel_w
        
        # Ensure dimensions are divisible
        assert H % h == 0 and W % w == 0, f"Frame size ({H},{W}) must be divisible by kernel size ({h},{w})"
        
        n = frames * H * W
        indices = []
        
        # Generate indices following spatial locality
        for f in range(frames):
            for i in range(H // h):
                for j in range(W // w):
                    for u in range(h):
                        for v in range(w):
                            y = i * h + u
                            x = j * w + v
                            idx = f * H * W + y * W + x
                            indices.append(idx)
        
        indices = torch.tensor(indices, dtype=torch.long, device=device)
        
        # Store reorder indices
        self.reorder_indices = indices
        
        # Generate restore indices (inverse permutation)
        restore = torch.empty_like(indices)
        restore[indices] = torch.arange(n, device=device)
        self.restore_indices = restore
        
    def average_pool_2d(self, x: torch.Tensor, frame_size: Tuple[int, int], frames: int) -> torch.Tensor:
        """
        Perform 2D average pooling over spatial dimensions.
        
        Args:
            x: Input tensor [B, L, D] where L = frames * H * W
            frame_size: (H, W) spatial dimensions
            frames: Number of temporal frames
            
        Returns:
            Pooled tensor [B, g, D] where g = frames * (H//h) * (W//w)
        """
        B, L, D = x.shape
        H, W = frame_size
        h, w = self.kernel_h, self.kernel_w
        
        # Reshape to [B, frames, H, W, D]
        x = x.view(B, frames, H, W, D)
        
        # Reshape for pooling: [B*frames, D, H, W]
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(B * frames, D, H, W)
        
        # Apply average pooling
        pooled = F.avg_pool2d(x, kernel_size=(h, w), stride=(h, w))
        
        # Reshape back: [B, g, D]
        _, D_pooled, H_pooled, W_pooled = pooled.shape
        pooled = pooled.view(B, frames, D, H_pooled, W_pooled)
        pooled = pooled.permute(0, 1, 3, 4, 2).contiguous()
        pooled = pooled.view(B, -1, D)
        
        return pooled
    
    def compute_sparsity_mask(self, draft_attn: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity mask from draft attention.
        
        Args:
            draft_attn: Draft attention scores [B, num_heads, g, g]
            
        Returns:
            Sparsity mask [B, num_heads, g, g]
        """
        B, num_heads, g, _ = draft_attn.shape
        
        # Flatten to compute top-k
        flat_attn = draft_attn.view(B * num_heads, -1)
        
        # Number of elements to keep
        k = int(self.sparsity_ratio * g * g)
        
        # Find threshold values
        top_k_values, _ = torch.topk(flat_attn, k, dim=-1)
        thresholds = top_k_values[:, -1].view(B, num_heads, 1, 1)
        
        # Create mask
        mask = (draft_attn >= thresholds).float()
        
        return mask
    
    def expand_mask_to_tokens(self, mask: torch.Tensor, frame_size: Tuple[int, int], frames: int) -> torch.Tensor:
        """
        Expand region-level mask to token-level mask.
        
        Args:
            mask: Region mask [B, num_heads, g, g]
            frame_size: (H, W) spatial dimensions
            frames: Number of temporal frames
            
        Returns:
            Token mask [B, num_heads, L, L] where L = frames * H * W
        """
        B, num_heads, g, _ = mask.shape
        H, W = frame_size
        h, w = self.kernel_h, self.kernel_w
        
        # Calculate tokens per region
        tokens_per_region = h * w
        L = frames * H * W
        
        # Expand mask by repeating each region's mask across tokens
        mask_expanded = mask.repeat_interleave(tokens_per_region, dim=-2)
        mask_expanded = mask_expanded.repeat_interleave(tokens_per_region, dim=-1)
        
        return mask_expanded
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        frame_size: Optional[Tuple[int, int]] = None,
        frames: int = 1,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DraftAttention.
        
        Args:
            hidden_states: Input tensor [B, L, D]
            frame_size: (H, W) spatial dimensions, inferred if not provided
            frames: Number of temporal frames
            attention_mask: Optional attention mask [B, L]
            
        Returns:
            Output tensor [B, L, D]
        """
        B, L, D = hidden_states.shape
        
        # Infer frame_size if not provided
        if frame_size is None:
            # Assume square frames
            spatial_tokens = L // frames
            H = W = int(math.sqrt(spatial_tokens))
            frame_size = (H, W)
            assert H * W == spatial_tokens, "Spatial tokens must form a square"
        
        H, W = frame_size
        
        # Generate reorder indices if not already done
        if self.reorder_indices is None or self.reorder_indices.device != hidden_states.device:
            self.generate_reorder_indices(frame_size, frames, hidden_states.device)
        
        # Reorder hidden states for spatial locality
        hidden_states_reordered = hidden_states[:, self.reorder_indices, :]
        
        # Project to Q, K, V
        Q = self.q_proj(hidden_states_reordered)
        K = self.k_proj(hidden_states_reordered)
        V = self.v_proj(hidden_states_reordered)
        
        # Reshape for multi-head attention
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Shape: [B, num_heads, L, head_dim]
        
        # Step 1: Compute draft attention
        Q_draft = self.average_pool_2d(Q.transpose(1, 2).contiguous().view(B, L, -1), frame_size, frames)
        K_draft = self.average_pool_2d(K.transpose(1, 2).contiguous().view(B, L, -1), frame_size, frames)
        
        # Reshape for multi-head: [B, g, num_heads * head_dim]
        g = Q_draft.shape[1]  # Number of regions
        Q_draft = Q_draft.view(B, g, self.num_heads, self.head_dim).transpose(1, 2)
        K_draft = K_draft.view(B, g, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute draft attention scores
        draft_scores = torch.matmul(Q_draft, K_draft.transpose(-2, -1)) / self.scale
        
        # Step 2: Compute sparsity mask
        mask_regions = self.compute_sparsity_mask(draft_scores)
        mask_tokens = self.expand_mask_to_tokens(mask_regions, frame_size, frames)
        
        # Step 3: Compute sparse attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply sparsity mask
        scores_masked = scores * mask_tokens
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention mask for all heads
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            scores_masked = scores_masked.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores_masked, dim=-1)
        
        # Apply dropout if training
        attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(attn_output)
        
        # Restore original order
        output_restored = torch.empty_like(output)
        output_restored[:, self.restore_indices, :] = output
        
        return output_restored
    
    def load_weights(self, state_dict: dict):
        """Load pre-trained weights."""
        self.load_state_dict(state_dict)
        
    def save_weights(self, path: str):
        """Save weights to file."""
        torch.save(self.state_dict(), path)