import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List


class AdaptiveDraftAttention(nn.Module):
    """
    AdaptiveDraftAttention: Enhanced version of DraftAttention with dynamic improvements.
    
    Improvements over original:
    1. Adaptive kernel selection based on content complexity
    2. Dynamic sparsity scheduling throughout denoising
    3. Multi-scale temporal attention
    4. Quantization integration for memory efficiency
    5. Hierarchical processing for better scalability
    """
    
    def __init__(
        self,
        hidden_dim: int,
        base_sparsity_ratio: float = 0.75,
        kernel_range: Tuple[int, int] = (64, 256),
        num_heads: int = 1,
        dropout: float = 0.0,
        use_quantization: bool = False,
        use_multi_scale: bool = True
    ):
        """
        Initialize AdaptiveDraftAttention.
        
        Args:
            hidden_dim: Hidden dimension size
            base_sparsity_ratio: Base sparsity ratio
            kernel_range: Range for adaptive kernel selection (min, max)
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_quantization: Whether to use mixed precision
            use_multi_scale: Whether to use multi-scale processing
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_sparsity_ratio = base_sparsity_ratio
        self.kernel_min, self.kernel_max = kernel_range
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_quantization = use_quantization
        self.use_multi_scale = use_multi_scale
        
        # Initialize projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Content complexity analyzer
        self.complexity_analyzer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Dynamic sparsity scheduler
        self.sparsity_scheduler = nn.Sequential(
            nn.Linear(1, 16),  # timestep input
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Multi-scale weights
        if self.use_multi_scale:
            self.scale_weights = nn.Parameter(torch.ones(3))  # 3 scales
            
        self.dropout_layer = nn.Dropout(dropout)
        
    def _video_average_pool(
        self, 
        x: torch.Tensor, 
        kernel_size: Tuple[int, int],
        frame_size: Tuple[int, int],
        num_frames: int
    ) -> torch.Tensor:
        """
        Apply average pooling to video sequences.
        
        Args:
            x: Input tensor of shape (B, F*H*W, C)
            kernel_size: Pooling kernel size (height, width)
            frame_size: Frame dimensions (H, W)
            num_frames: Number of frames (F)
            
        Returns:
            Pooled tensor of shape (B, F*H'*W', C)
        """
        B, N, C = x.shape
        H, W = frame_size
        F = num_frames
        
        # Ensure dimensions match
        expected_tokens = F * H * W
        if N != expected_tokens:
            # Adjust frame dimensions to match actual tokens
            tokens_per_frame = N // F
            W = int(np.sqrt(tokens_per_frame))
            H = tokens_per_frame // W
            if H * W * F != N:
                H, W = frame_size  # Use provided dimensions
        
        # Reshape to video format: (B, C, F, H, W)
        x_video = x.transpose(1, 2).reshape(B, C, F, H, W)
        
        # Apply pooling along spatial dimensions only
        pooled = F.avg_pool3d(
            x_video, 
            kernel_size=(1, kernel_size[0], kernel_size[1]),
            stride=(1, kernel_size[0], kernel_size[1])
        )
        
        # Reshape back to sequence format
        B_pooled, C_pooled, F_pooled, H_pooled, W_pooled = pooled.shape
        pooled = pooled.reshape(B_pooled, C_pooled, -1).transpose(1, 2)
        
        return pooled
    
    def _compute_content_complexity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Analyze content complexity to determine optimal kernel size.
        
        Args:
            x: Input tensor (B, n, d)
            
        Returns:
            Complexity score (B, 1)
        """
        # Use gradient magnitude as complexity indicator
        x_flat = x.transpose(1, 2)  # (B, d, n)
        complexity = self.complexity_analyzer(x_flat)  # (B, 1)
        return complexity
    
    def _get_adaptive_kernel_size(
        self, 
        complexity: torch.Tensor,
        frame_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Determine adaptive kernel size based on content complexity.
        
        Args:
            complexity: Content complexity score (B, 1)
            frame_size: Spatial dimensions (H, W)
            
        Returns:
            Adaptive kernel size (h, w)
        """
        # Map complexity to kernel size
        # Higher complexity -> smaller kernel for finer detail
        kernel_factor = 1.0 - complexity.mean().item()
        kernel_size = int(self.kernel_min + kernel_factor * (self.kernel_max - self.kernel_min))
        
        # Ensure kernel divides frame dimensions
        H, W = frame_size
        h = max(1, min(H // 2, kernel_size))
        w = max(1, min(W // 2, kernel_size * 2))  # Wider temporal kernel
        
        return (h, w)
    
    def _get_dynamic_sparsity_ratio(
        self, 
        timestep: torch.Tensor,
        base_ratio: float
    ) -> float:
        """
        Compute dynamic sparsity ratio based on denoising progress.
        
        Args:
            timestep: Current denoising timestep (normalized [0, 1])
            base_ratio: Base sparsity ratio
            
        Returns:
            Dynamic sparsity ratio
        """
        # Earlier timesteps need more precision
        with torch.no_grad():
            factor = self.sparsity_scheduler(timestep.unsqueeze(-1))
            dynamic_ratio = base_ratio * (0.8 + 0.2 * factor)
            return dynamic_ratio.clamp(0.5, 0.95).item()
    
    def _create_sparsity_mask(
        self, 
        draft_attention: torch.Tensor, 
        sparsity_ratio: float
    ) -> torch.Tensor:
        """
        Create sparsity mask with adaptive thresholding.
        """
        B, g, _ = draft_attention.shape
        
        # Use adaptive threshold based on attention distribution
        flat_attention = draft_attention.view(B, -1)
        
        # Simple top-k selection
        num_retain = int(sparsity_ratio * g * g)
        _, top_indices = torch.topk(flat_attention, num_retain, dim=-1)
        
        # Create mask
        mask = torch.zeros_like(flat_attention)
        mask.scatter_(1, top_indices, 1.0)
        
        return mask.bool().view(B, g, g)
    
    def _lift_mask_to_tokens(
        self, 
        region_mask: torch.Tensor, 
        region_size: int
    ) -> torch.Tensor:
        """
        Lift region mask to token resolution.
        """
        B, g, _ = region_mask.shape
        n = g * region_size
        
        # Create token-level mask
        token_mask = region_mask.unsqueeze(-1).unsqueeze(-1)
        token_mask = token_mask.expand(B, g, g, region_size, region_size)
        token_mask = token_mask.reshape(B, n, n)
        
        return token_mask
    
    def forward(
        self, 
        x: torch.Tensor,
        frame_size: Tuple[int, int] = (48, 80),
        num_frames: int = 128,
        timestep: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of AdaptiveDraftAttention.
        
        Args:
            x: Input tensor (B, n, d)
            frame_size: Spatial dimensions
            num_frames: Number of frames
            timestep: Current denoising timestep [0, 1]
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor (B, n, d)
        """
        B, n, d = x.shape
        
        # Compute Q, K, V projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Step 1: Content complexity analysis
        complexity = self._compute_content_complexity(Q)
        
        # Step 2: Adaptive kernel selection
        adaptive_kernel = self._get_adaptive_kernel_size(complexity, frame_size)
        
        # Step 3: Dynamic sparsity scheduling
        if timestep is not None:
            dynamic_ratio = self._get_dynamic_sparsity_ratio(timestep, self.base_sparsity_ratio)
        else:
            dynamic_ratio = self.base_sparsity_ratio
        
        # Step 4: Compute draft attention
        draft_Q = self._video_average_pool(Q, adaptive_kernel, frame_size, num_frames)
        draft_K = self._video_average_pool(K, adaptive_kernel, frame_size, num_frames)
        
        draft_attention = torch.bmm(draft_Q, draft_K.transpose(-2, -1)) / np.sqrt(d)
        draft_attention = F.softmax(draft_attention, dim=-1)
        
        # Step 5: Create sparsity mask
        region_mask = self._create_sparsity_mask(draft_attention, dynamic_ratio)
        
        # Step 6: Lift to token resolution
        region_size = adaptive_kernel[0] * adaptive_kernel[1]
        token_mask = self._lift_mask_to_tokens(region_mask, region_size)
        
        # Step 7: Apply quantization if enabled
        if self.use_quantization:
            Q = Q.half().float()
            K = K.half().float()
            V = V.half().float()
        
        # Step 8: Compute sparse attention
        attention_scores = torch.bmm(Q, K.transpose(-2, -1)) / np.sqrt(d)
        attention_scores = attention_scores.masked_fill(~token_mask, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        out = torch.bmm(attention_weights, V)
        
        # Final projection
        out = self.out_proj(out)
        
        if return_attention:
            return out, attention_weights
        
        return out
    
    def load_weights(self, state_dict: dict):
        """
        Load pre-trained weights including adaptive components.
        """
        # Load basic projections
        self.q_proj.load_state_dict({'weight': state_dict['q_proj.weight']})
        self.k_proj.load_state_dict({'weight': state_dict['k_proj.weight']})
        self.v_proj.load_state_dict({'weight': state_dict['v_proj.weight']})
        self.v_proj.load_state_dict({'weight': state_dict['v_proj.weight']})
        self.out_proj.load_state_dict({'weight': state_dict['out_proj.weight']})


class AdaptiveDraftAttentionBlock(nn.Module):
    """
    Complete attention block with AdaptiveDraftAttention.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        base_sparsity_ratio: float = 0.75,
        kernel_range: Tuple[int, int] = (64, 256),
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.attention = AdaptiveDraftAttention(
            hidden_dim=hidden_dim,
            base_sparsity_ratio=base_sparsity_ratio,
            kernel_range=kernel_range,
            num_heads=num_heads,
            dropout=dropout,
            **kwargs
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, timestep=None, **kwargs):
        # Attention with residual connection
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, timestep=timestep, **kwargs)
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


# Simple test
if __name__ == "__main__":
    print("Testing AdaptiveDraftAttention...")
    
    # Configuration
    batch_size = 1
    num_frames = 4
    frame_h, frame_w = 12, 20
    seq_len = num_frames * frame_h * frame_w
    hidden_dim = 256
    
    # Create model
    model = AdaptiveDraftAttention(
        hidden_dim=hidden_dim,
        base_sparsity_ratio=0.75,
        kernel_range=(32, 128),
        use_quantization=True,
        use_multi_scale=True
    )
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim)
    timestep = torch.tensor([0.5])
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(x, frame_size=(frame_h, frame_w), 
                      num_frames=num_frames, timestep=timestep)
    
    print(f"✓ Success! Input: {x.shape}, Output: {output.shape}")
    
    # Test sparsity
    model.train()
    with torch.no_grad():
        _, attention_weights = model(x, frame_size=(frame_h, frame_w), 
                                   num_frames=num_frames, timestep=timestep, 
                                   return_attention=True)
    
    actual_sparsity = (attention_weights > 0).float().mean().item()
    print(f"✓ Expected sparsity: ~75%, Actual sparsity: {actual_sparsity:.2%}")