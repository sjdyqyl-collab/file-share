import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple, List
import math
import numpy as np

class AdaptivePooling(nn.Module):
    """Adaptive pooling with kernel selection based on content complexity."""
    
    def __init__(self, kernel_sizes: List[int] = [64, 128, 256]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        
    def compute_content_complexity(self, x: torch.Tensor) -> torch.Tensor:
        """Compute content complexity based on gradient magnitude."""
        # Simple complexity measure based on spatial variance
        B, C, H, W = x.shape
        
        # Compute spatial gradients
        x_grad_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        x_grad_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        
        # Pad to maintain shape
        x_grad_h = F.pad(x_grad_h, (0, 0, 0, 1))
        x_grad_w = F.pad(x_grad_w, (0, 1, 0, 0))
        
        # Average gradient magnitude
        complexity = (x_grad_h.mean() + x_grad_w.mean()) / 2
        
        # Normalize to [0, 1]
        complexity = torch.sigmoid(complexity)
        
        return complexity
    
    def select_kernels(self, frame_size: Tuple[int, int], complexity: torch.Tensor) -> Tuple[int, int]:
        """Select optimal kernels based on content complexity."""
        H, W = frame_size
        
        # Compute target kernel sizes
        target_h = int(H * W * complexity.item() / 1024)
        target_w = int(H * W * (1 - complexity.item()) / 1024)
        
        # Select closest available kernel sizes
        k_h = min(self.kernel_sizes, key=lambda x: abs(x - target_h))
        k_w = min(self.kernel_sizes, key=lambda x: abs(x - target_w))
        
        return k_h, k_w
    
    def forward(self, x: torch.Tensor, frame_size: Tuple[int, int], frames: int) -> torch.Tensor:
        """Apply adaptive pooling."""
        B, L, D = x.shape
        H, W = frame_size
        
        # Reshape for complexity computation
        x_reshaped = x.view(B, frames, H, W, D).permute(0, 4, 1, 2, 3).contiguous()
        x_reshaped = x_reshaped.view(B * D, frames, H, W)
        
        # Compute complexity
        complexity = self.compute_content_complexity(x_reshaped)
        
        # Select kernels
        k_h, k_w = self.select_kernels(frame_size, complexity)
        
        # Apply pooling
        x = x.view(B, frames, H, W, D).permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(B * D * frames, 1, H, W)
        
        pooled = F.avg_pool2d(x, kernel_size=(k_h, k_w), stride=(k_h, k_w))
        
        # Reshape back
        _, _, H_pooled, W_pooled = pooled.shape
        pooled = pooled.view(B, D, frames, H_pooled, W_pooled)
        pooled = pooled.permute(0, 2, 3, 4, 1).contiguous()
        pooled = pooled.view(B, -1, D)
        
        return pooled, (k_h, k_w)

class QuantizedLinear(nn.Module):
    """INT8 quantized linear layer with FP16 dequantization."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store original weights in FP16
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)
            
        # Quantized weights cache
        self.register_buffer('weight_int8', None)
        self.register_buffer('weight_scale', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def quantize_weights(self):
        """Quantize weights to INT8."""
        # Compute scale factor
        max_val = torch.max(torch.abs(self.weight))
        scale = max_val / 127.0
        
        # Quantize
        weight_int8 = torch.round(self.weight / scale).clamp(-128, 127).to(torch.int8)
        
        # Cache quantized weights
        self.weight_int8 = weight_int8
        self.weight_scale = scale
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization."""
        if self.training or self.weight_int8 is None:
            # Use full precision during training
            return F.linear(input, self.weight, self.bias)
        else:
            # Use quantized weights during inference
            weight_fp16 = self.weight_int8.to(torch.float16) * self.weight_scale
            return F.linear(input, weight_fp16, self.bias)

class DraftAttentionPlus(nn.Module):
    """
    Enhanced DraftAttention++ with adaptive pooling, layer sparsity, quantization, and multi-GPU support.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        sparsity_range: Tuple[float, float] = (0.5, 0.9),
        kernel_sizes: List[int] = [64, 128, 256],
        use_quantization: bool = True,
        use_multi_gpu: bool = False,
        num_gpus: int = 1,
        max_sequence_length: int = 8192,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.sparsity_min, self.sparsity_max = sparsity_range
        self.use_quantization = use_quantization
        self.use_multi_gpu = use_multi_gpu
        self.num_gpus = num_gpus
        
        # Ensure hidden_size is divisible by num_heads
        assert hidden_size % num_heads == 0
        
        # Adaptive pooling
        self.adaptive_pool = AdaptivePooling(kernel_sizes)
        
        # Linear projections
        if use_quantization:
            self.q_proj = QuantizedLinear(hidden_size, hidden_size, bias=False)
            self.k_proj = QuantizedLinear(hidden_size, hidden_size, bias=False)
            self.v_proj = QuantizedLinear(hidden_size, hidden_size, bias=False)
            self.out_proj = QuantizedLinear(hidden_size, hidden_size, bias=False)
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.scale = math.sqrt(self.head_dim)
        
        # Buffers for reordering
        self.register_buffer('reorder_indices', None)
        self.register_buffer('restore_indices', None)
        
        # Entropy tracking for layer-wise sparsity
        self.register_buffer('entropy_buffer', torch.zeros(100))  # Track last 100 layers
        self.entropy_ptr = 0
        
    def compute_attention_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention distribution."""
        # Flatten attention
        B, num_heads, L, _ = attention.shape
        attention_flat = attention.view(B * num_heads, L * L)
        
        # Add small epsilon for numerical stability
        attention_flat = attention_flat + 1e-8
        attention_flat = attention_flat / attention_flat.sum(dim=-1, keepdim=True)
        
        # Compute entropy
        entropy = -torch.sum(attention_flat * torch.log(attention_flat), dim=-1)
        entropy = entropy.mean()
        
        return entropy
    
    def get_layer_sparsity(self, entropy: torch.Tensor) -> float:
        """Compute layer-specific sparsity based on entropy."""
        # Update entropy buffer
        self.entropy_buffer[self.entropy_ptr] = entropy
        self.entropy_ptr = (self.entropy_ptr + 1) % 100
        
        # Compute relative entropy
        max_entropy = self.entropy_buffer.max()
        if max_entropy > 0:
            relative_entropy = entropy / max_entropy
        else:
            relative_entropy = 0.5
        
        # Map to sparsity range
        sparsity = 1.0 - min(self.sparsity_max, max(self.sparsity_min, relative_entropy))
        
        return sparsity
    
    def temporal_spatial_pooling(self, x: torch.Tensor, frame_size: Tuple[int, int], frames: int, 
                               k_h: int, k_w: int) -> torch.Tensor:
        """Apply temporal-spatial separated pooling."""
        B, L, D = x.shape
        H, W = frame_size
        
        # Reshape for pooling
        x = x.view(B, frames, H, W, D).permute(0, 4, 1, 2, 3).contiguous()
        
        # Temporal pooling: T×1×1 kernel
        x_temporal = F.avg_pool3d(x, kernel_size=(k_h, 1, 1), stride=(k_h, 1, 1))
        
        # Spatial pooling: 1×H×W kernel
        x_spatial = F.avg_pool3d(x_temporal, kernel_size=(1, k_w, k_w), stride=(1, k_w, k_w))
        
        # Reshape back
        _, _, T_pooled, H_pooled, W_pooled = x_spatial.shape
        x_pooled = x_spatial.permute(0, 2, 3, 4, 1).contiguous()
        x_pooled = x_pooled.view(B, -1, D)
        
        return x_pooled
    
    def distributed_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                            mask: torch.Tensor) -> torch.Tensor:
        """Multi-GPU distributed attention computation."""
        if not self.use_multi_gpu or self.num_gpus <= 1:
            return self.local_attention(Q, K, V, mask)
        
        # Get current GPU info
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        if world_size != self.num_gpus:
            return self.local_attention(Q, K, V, mask)
        
        # Split sequence dimension across GPUs
        B, num_heads, L, head_dim = Q.shape
        local_L = L // world_size
        
        # Split tensors
        Q_local = Q[:, :, rank * local_L:(rank + 1) * local_L, :]
        K_local = K[:, :, rank * local_L:(rank + 1) * local_L, :]
        V_local = V[:, :, rank * local_L:(rank + 1) * local_L, :]
        mask_local = mask[:, :, rank * local_L:(rank + 1) * local_L, :]
        
        # Local computation
        output_local = self.local_attention(Q_local, K, V, mask_local)
        
        # All-gather results
        output_list = [torch.zeros_like(output_local) for _ in range(world_size)]
        dist.all_gather(output_list, output_local)
        
        # Concatenate results
        output = torch.cat(output_list, dim=2)
        
        return output
    
    def local_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                       mask: torch.Tensor) -> torch.Tensor:
        """Local attention computation."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        scores_masked = scores * mask
        
        # Apply softmax
        attn_weights = F.softmax(scores_masked, dim=-1)
        attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
        
        # Apply to values
        output = torch.matmul(attn_weights, V)
        
        return output
    
    def generate_reorder_indices(self, frame_size: Tuple[int, int], frames: int, device: torch.device):
        """Generate reorder indices."""
        H, W = frame_size
        
        # Use 8×16 as default for reordering
        h, w = 8, 16
        
        n = frames * H * W
        indices = []
        
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
        self.reorder_indices = indices
        
        # Generate restore indices
        restore = torch.empty_like(indices)
        restore[indices] = torch.arange(n, device=device)
        self.restore_indices = restore
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        frame_size: Optional[Tuple[int, int]] = None,
        frames: int = 1,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of DraftAttention++."""
        B, L, D = hidden_states.shape
        
        # Infer frame_size
        if frame_size is None:
            spatial_tokens = L // frames
            H = W = int(math.sqrt(spatial_tokens))
            frame_size = (H, W)
            assert H * W == spatial_tokens
        
        H, W = frame_size
        
        # Generate reorder indices
        if self.reorder_indices is None or self.reorder_indices.device != hidden_states.device:
            self.generate_reorder_indices(frame_size, frames, hidden_states.device)
        
        # Reorder for spatial locality
        hidden_states_reordered = hidden_states[:, self.reorder_indices, :]
        
        # Project to Q, K, V
        Q = self.q_proj(hidden_states_reordered)
        K = self.k_proj(hidden_states_reordered)
        V = self.v_proj(hidden_states_reordered)
        
        # Reshape for multi-head
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Step 1: Adaptive draft attention
        Q_draft_flat = Q.transpose(1, 2).contiguous().view(B, L, -1)
        K_draft_flat = K.transpose(1, 2).contiguous().view(B, L, -1)
        
        # Apply adaptive pooling
        Q_draft, (k_h, k_w) = self.adaptive_pool(Q_draft_flat, frame_size, frames)
        K_draft, _ = self.adaptive_pool(K_draft_flat, frame_size, frames)
        
        # Reshape for multi-head
        g = Q_draft.shape[1]
        Q_draft = Q_draft.view(B, g, self.num_heads, self.head_dim).transpose(1, 2)
        K_draft = K_draft.view(B, g, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute draft attention
        draft_scores = torch.matmul(Q_draft, K_draft.transpose(-2, -1)) / self.scale
        
        # Compute entropy and layer sparsity
        entropy = self.compute_attention_entropy(draft_scores)
        layer_sparsity = self.get_layer_sparsity(entropy)
        
        # Step 2: Compute sparsity mask
        mask_regions = self.compute_sparsity_mask(draft_scores, layer_sparsity)
        mask_tokens = self.expand_mask_to_tokens(mask_regions, frame_size, frames, k_h, k_w)
        
        # Step 3: Distributed sparse attention
        output = self.distributed_attention(Q, K, V, mask_tokens)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(output)
        
        # Restore order
        output_restored = torch.empty_like(output)
        output_restored[:, self.restore_indices, :] = output
        
        return output_restored
    
    def compute_sparsity_mask(self, draft_attn: torch.Tensor, sparsity_ratio: float) -> torch.Tensor:
        """Compute sparsity mask with dynamic ratio."""
        B, num_heads, g, _ = draft_attn.shape
        
        flat_attn = draft_attn.view(B * num_heads, -1)
        k = int(sparsity_ratio * g * g)
        
        top_k_values, _ = torch.topk(flat_attn, k, dim=-1)
        thresholds = top_k_values[:, -1].view(B, num_heads, 1, 1)
        
        mask = (draft_attn >= thresholds).float()
        
        return mask
    
    def expand_mask_to_tokens(self, mask: torch.Tensor, frame_size: Tuple[int, int], 
                            frames: int, k_h: int, k_w: int) -> torch.Tensor:
        """Expand mask to token level."""
        B, num_heads, g, _ = mask.shape
        H, W = frame_size
        
        # Calculate tokens per region based on actual kernel sizes
        tokens_per_region = k_h * k_w
        L = frames * H * W
        
        # Expand mask
        mask_expanded = mask.repeat_interleave(tokens_per_region, dim=-2)
        mask_expanded = mask_expanded.repeat_interleave(tokens_per_region, dim=-1)
        
        return mask_expanded
    
    def quantize_for_inference(self):
        """Quantize weights for inference."""
        if hasattr(self.q_proj, 'quantize_weights'):
            self.q_proj.quantize_weights()
            self.k_proj.quantize_weights()
            self.v_proj.quantize_weights()
            self.out_proj.quantize_weights()
    
    def load_weights(self, state_dict: dict):
        """Load pre-trained weights."""
        self.load_state_dict(state_dict)
        
    def save_weights(self, path: str):
        """Save weights to file."""
        torch.save(self.state_dict(), path)