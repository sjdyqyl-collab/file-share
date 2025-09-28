import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class DraftAttention(nn.Module):
    """
    DraftAttention: Training-free dynamic sparse attention using low-resolution draft maps
    
    Paper: "DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance"
    
    This implementation provides:
    1. Low-resolution draft attention computation via average pooling
    2. Hardware-efficient token reordering
    3. Dynamic sparse attention based on draft guidance
    4. Training-free acceleration with quality preservation
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        pooling_kernel: Tuple[int, int] = (8, 16),
        sparsity_ratio: float = 0.9,
        frame_height: int = 48,
        frame_width: int = 80,
        max_frames: int = 128,
    ):
        """
        Initialize DraftAttention module.
        
        Args:
            hidden_size: Hidden dimension size (d)
            num_heads: Number of attention heads
            pooling_kernel: (height, width) pooling kernel for draft attention
            sparsity_ratio: Target sparsity ratio (fraction to keep)
            frame_height: Height of feature map in tokens
            frame_width: Width of feature map in tokens  
            max_frames: Maximum number of frames
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.pooling_kernel = pooling_kernel
        self.sparsity_ratio = sparsity_ratio
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.max_frames = max_frames
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Calculate reduction factor
        self.reduction_h = pooling_kernel[0]
        self.reduction_w = pooling_kernel[1]
        self.reduction_factor = self.reduction_h * self.reduction_w  # 128 for 8x16
        
        # Initialize projection layers
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Pre-compute reordering indices for efficiency
        self.register_buffer('reorder_indices', None)
        self.register_buffer('inverse_reorder_indices', None)
        
    def _compute_reorder_indices(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Compute reordering indices for hardware-efficient memory access.
        
        Groups tokens into contiguous 8x16 patches for coalesced GPU access.
        
        Args:
            batch_size: Batch size B
            seq_len: Sequence length L
            
        Returns:
            reorder_indices: Tensor of shape [L] containing reordering indices
        """
        device = self.reorder_indices.device if self.reorder_indices is not None else 'cpu'
        
        # Calculate spatial dimensions
        tokens_per_frame = self.frame_height * self.frame_width
        num_frames = seq_len // tokens_per_frame
        
        # Create coordinate grids
        h_coords = torch.arange(self.frame_height, device=device)
        w_coords = torch.arange(self.frame_width, device=device)
        
        # Create patch-based reordering
        indices = []
        for f in range(num_frames):
            for ph in range(0, self.frame_height, self.reduction_h):
                for pw in range(0, self.frame_width, self.reduction_w):
                    # Extract 8x16 patch indices
                    patch_indices = []
                    for h in range(ph, min(ph + self.reduction_h, self.frame_height)):
                        for w in range(pw, min(pw + self.reduction_w, self.frame_width)):
                            idx = f * tokens_per_frame + h * self.frame_width + w
                            patch_indices.append(idx)
                    
                    # Pad patch to fixed size if needed
                    while len(patch_indices) < self.reduction_factor:
                        patch_indices.append(patch_indices[-1])  # Repeat last token
                    
                    indices.extend(patch_indices)
        
        reorder_indices = torch.tensor(indices[:seq_len], device=device, dtype=torch.long)
        
        # Compute inverse indices for restoration
        inverse_indices = torch.empty_like(reorder_indices)
        inverse_indices[reorder_indices] = torch.arange(seq_len, device=device)
        
        return reorder_indices, inverse_indices
    
    def _create_draft_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor
    ) -> torch.Tensor:
        """
        Create low-resolution draft attention map via average pooling.
        
        Args:
            q: Query tensor [B, L, H]
            k: Key tensor [B, L, H]
            
        Returns:
            draft_attention: Draft attention map [B, g, g] where g = L/128
        """
        B, L, H = q.shape
        
        # Calculate spatial dimensions for pooling
        tokens_per_frame = self.frame_height * self.frame_width
        num_frames = L // tokens_per_frame
        
        # Reshape for pooling: [B, F, H_frame, W_frame, H]
        q_reshaped = q.view(B, num_frames, self.frame_height, self.frame_width, H)
        k_reshaped = k.view(B, num_frames, self.frame_height, self.frame_width, H)
        
        # Transpose for pooling: [B, F, H, H_frame, W_frame]
        q_reshaped = q_reshaped.permute(0, 1, 4, 2, 3)  # [B, F, H, H_frame, W_frame]
        k_reshaped = k_reshaped.permute(0, 1, 4, 2, 3)
        
        # Average pooling over spatial dimensions
        pool = nn.AvgPool2d(kernel_size=self.pooling_kernel, stride=self.pooling_kernel)
        
        q_draft = pool(q_reshaped.flatten(0, 1))  # [B*F, H, H_draft, W_draft]
        k_draft = pool(k_reshaped.flatten(0, 1))  # [B*F, H, H_draft, W_draft]
        
        # Reshape back: [B, F, H, H_draft*W_draft]
        B_draft, H_draft, Hg, Wg = q_draft.shape
        q_draft = q_draft.view(B, num_frames, H, -1).transpose(-2, -1)  # [B, F, g, H]
        k_draft = k_draft.view(B, num_frames, H, -1).transpose(-2, -1)  # [B, F, g, H]
        
        # Flatten frames and spatial dimensions
        q_draft = q_draft.flatten(1, 2)  # [B, F*g, H]
        k_draft = k_draft.flatten(1, 2)  # [B, F*g, H]
        
        # Compute draft attention
        scale = 1.0 / np.sqrt(H)
        draft_attention = torch.bmm(q_draft, k_draft.transpose(-2, -1)) * scale
        draft_attention = F.softmax(draft_attention, dim=-1)
        
        return draft_attention
    
    def _create_sparsity_mask(
        self, 
        draft_attention: torch.Tensor, 
        full_seq_len: int
    ) -> torch.Tensor:
        """
        Create sparsity mask based on draft attention scores.
        
        Args:
            draft_attention: Draft attention map [B, g, g]
            full_seq_len: Original sequence length L
            
        Returns:
            sparsity_mask: Binary mask [B, L, L] indicating which connections to keep
        """
        B, g, _ = draft_attention.shape
        
        # Calculate mapping from draft indices to full indices
        reduction_factor = self.reduction_factor
        full_g = full_seq_len // reduction_factor
        
        # Ensure dimensions match
        if g != full_g:
            # Pad or crop draft attention to match expected size
            target_g = full_seq_len // reduction_factor
            if g < target_g:
                # Pad with zeros
                pad_size = target_g - g
                draft_attention = F.pad(draft_attention, (0, pad_size, 0, pad_size))
            else:
                # Crop
                draft_attention = draft_attention[:, :target_g, :target_g]
            g = draft_attention.shape[1]
        
        # Create block-wise sparsity mask
        mask = torch.zeros(B, full_seq_len, full_seq_len, device=draft_attention.device)
        
        # Determine number of blocks to keep based on sparsity ratio
        total_blocks = g * g
        keep_blocks = int(total_blocks * (1 - self.sparsity_ratio))
        
        for b in range(B):
            # Flatten draft attention and get top-k indices
            flat_scores = draft_attention[b].flatten()
            _, top_indices = torch.topk(flat_scores, k=keep_blocks)
            
            # Convert flat indices back to 2D
            row_indices = top_indices // g
            col_indices = top_indices % g
            
            # Set corresponding blocks in the mask
            for r, c in zip(row_indices, col_indices):
                start_r = r * reduction_factor
                end_r = min((r + 1) * reduction_factor, full_seq_len)
                start_c = c * reduction_factor
                end_c = min((c + 1) * reduction_factor, full_seq_len)
                
                mask[b, start_r:end_r, start_c:end_c] = 1.0
        
        return mask.bool()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of DraftAttention.
        
        Args:
            hidden_states: Input tensor [B, L, D]
            attention_mask: Optional attention mask [B, L]
            past_key_value: Optional cached key/value states
            output_attentions: Whether to return attention weights
            
        Returns:
            output: Output tensor [B, L, D]
            attention_weights: Optional attention weights [B, L, L]
        """
        B, L, D = hidden_states.shape
        
        # Ensure reorder indices are computed
        if self.reorder_indices is None or len(self.reorder_indices) != L:
            self.reorder_indices, self.inverse_reorder_indices = self._compute_reorder_indices(B, L)
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)  # [B, L, D]
        k = self.k_proj(hidden_states)  # [B, L, D]
        v = self.v_proj(hidden_states)  # [B, L, D]
        
        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        
        # Apply reordering for hardware efficiency
        q_reordered = q[:, :, self.reorder_indices, :]
        k_reordered = k[:, :, self.reorder_indices, :]
        v_reordered = v[:, :, self.reorder_indices, :]
        
        # Create draft attention for sparsity guidance
        # Average across heads for draft computation
        q_avg = q_reordered.mean(dim=1)  # [B, L, head_dim]
        k_avg = k_reordered.mean(dim=1)  # [B, L, head_dim]
        
        draft_attention = self._create_draft_attention(q_avg, k_avg)  # [B, g, g]
        
        # Create sparsity mask
        sparsity_mask = self._create_sparsity_mask(draft_attention, L)  # [B, L, L]
        
        # Expand mask for all heads
        sparsity_mask = sparsity_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [B, num_heads, L, L]
        
        # Compute attention scores
        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = torch.matmul(q_reordered, k_reordered.transpose(-2, -1)) * scale  # [B, num_heads, L, L]
        
        # Apply sparsity mask
        attn_scores = attn_scores.masked_fill(~sparsity_mask, float('-inf'))
        
        # Apply causal mask if needed (for autoregressive generation)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask.unsqueeze(1).unsqueeze(1)
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v_reordered)  # [B, num_heads, L, head_dim]
        
        # Restore original order
        attn_output_restored = torch.empty_like(attn_output)
        attn_output_restored[:, :, self.inverse_reorder_indices, :] = attn_output
        
        # Reshape and project output
        attn_output = attn_output_restored.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        output = self.out_proj(attn_output)
        
        if output_attentions:
            # Create full attention matrix for output (mostly zeros due to sparsity)
            full_attn = torch.zeros(B, self.num_heads, L, L, device=hidden_states.device)
            full_attn[:, :, self.reorder_indices, :] = attn_weights
            full_attn = full_attn[:, :, self.inverse_reorder_indices, :]
            full_attn = full_attn.mean(dim=1)  # Average over heads
            return output, full_attn
        
        return output, None
    
    def load_weights(self, state_dict: dict):
        """Load pre-trained weights into the module."""
        self.load_state_dict(state_dict)
    
    def get_sparsity_stats(self) -> dict:
        """Get statistics about current sparsity patterns."""
        return {
            'sparsity_ratio': self.sparsity_ratio,
            'reduction_factor': self.reduction_factor,
            'pooling_kernel': self.pooling_kernel,
        }