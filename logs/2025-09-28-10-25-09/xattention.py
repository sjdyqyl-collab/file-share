import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
import numpy as np

class XAttention(nn.Module):
    """
    XAttention: Efficient block-sparse attention using antidiagonal scoring.
    
    Shape conventions:
    - B: batch size
    - L: sequence length
    - H: hidden size (d_model)
    - num_heads: number of attention heads
    - head_dim: dimension per head (H // num_heads)
    - B_block: block size for sparse attention
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int = 64,
        stride: int = 8,
        threshold: float = 0.9,
        dropout: float = 0.1,
        use_dynamic_threshold: bool = True
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # XAttention parameters
        self.block_size = block_size
        self.stride = stride
        self.threshold = threshold
        self.use_dynamic_threshold = use_dynamic_threshold
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Dynamic threshold parameters
        if use_dynamic_threshold:
            self.register_buffer('head_thresholds', torch.ones(num_heads) * threshold)
            self.threshold_adjustments = 1000
            self.threshold_decay = 0.9
    
    def extract_antidiagonal_elements(self, tensor: torch.Tensor, block_idx: int) -> torch.Tensor:
        """
        Extract antidiagonal elements from a block with stride sampling.
        
        Args:
            tensor: [B, L, head_dim]
            block_idx: index of current block
            
        Returns:
            antidiagonal_features: [B, antidiagonal_len, head_dim]
        """
        B, L, D = tensor.shape
        start = block_idx * self.block_size
        end = min((block_idx + 1) * self.block_size, L)
        
        if end - start < self.block_size:
            # Pad if necessary
            pad_size = self.block_size - (end - start)
            block_tensor = F.pad(tensor[:, start:end, :], (0, 0, 0, pad_size))
        else:
            block_tensor = tensor[:, start:end, :]  # [B, block_size, head_dim]
        
        # Extract antidiagonal elements with stride
        B_block, L_block, D = block_tensor.shape
        antidiagonal_features = []
        
        for b in range(B):
            block = block_tensor[b]  # [block_size, head_dim]
            antidiags = []
            
            # Extract antidiagonal elements with stride
            for k in range(0, 2 * self.block_size - 1, self.stride):
                diag_elements = []
                for i in range(max(0, k - self.block_size + 1), 
                              min(k + 1, self.block_size)):
                    j = k - i
                    if 0 <= j < self.block_size:
                        diag_elements.append(block[i, :])
                
                if diag_elements:
                    # Average along antidiagonal
                    diag_tensor = torch.stack(diag_elements).mean(dim=0)
                    antidiags.append(diag_tensor)
            
            if antidiags:
                antidiagonal_features.append(torch.stack(antidiags))
            else:
                antidiagonal_features.append(torch.zeros(1, D, device=tensor.device))
        
        # Pad or truncate to consistent size
        max_len = max([f.shape[0] for f in antidiagonal_features])
        padded_features = []
        
        for f in antidiagonal_features:
            if f.shape[0] < max_len:
                pad_size = max_len - f.shape[0]
                padded = F.pad(f, (0, 0, 0, pad_size))
            else:
                padded = f[:max_len]
            padded_features.append(padded)
        
        return torch.stack(padded_features)  # [B, antidiagonal_len, head_dim]
    
    def compute_antidiagonal_scores(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Compute antidiagonal scores for block importance prediction.
        
        Args:
            Q: [B, num_heads, L, head_dim]
            K: [B, num_heads, L, head_dim]
            
        Returns:
            scores: [B, num_heads, num_blocks]
        """
        B, num_heads, L, head_dim = Q.shape
        num_blocks = (L + self.block_size - 1) // self.block_size
        
        scores = torch.zeros(B, num_heads, num_blocks, device=Q.device)
        
        for head in range(num_heads):
            Q_head = Q[:, head, :, :]  # [B, L, head_dim]
            K_head = K[:, head, :, :]  # [B, L, head_dim]
            
            for block_idx in range(num_blocks):
                # Extract antidiagonal elements
                Q_antidiag = self.extract_antidiagonal_elements(Q_head, block_idx)
                K_antidiag = self.extract_antidiagonal_elements(K_head, block_idx)
                
                # Compute approximate attention scores
                # Q_antidiag: [B, antidiagonal_len, head_dim]
                # K_antidiag: [B, antidiagonal_len, head_dim]
                
                if Q_antidiag.shape[1] > 0 and K_antidiag.shape[1] > 0:
                    # Compute attention approximation
                    attn_approx = torch.bmm(
                        Q_antidiag, K_antidiag.transpose(-2, -1)
                    ) / math.sqrt(head_dim * self.stride)  # [B, antidiagonal_len, antidiagonal_len]
                    
                    # Sum as importance score
                    score = attn_approx.sum(dim=(-2, -1))  # [B]
                    scores[:, head, block_idx] = score
        
        return scores
    
    def select_blocks(self, scores: torch.Tensor, threshold: float) -> List[List[torch.Tensor]]:
        """
        Select important blocks using threshold-based selection.
        
        Args:
            scores: [B, num_heads, num_blocks]
            threshold: selection threshold
            
        Returns:
            block_masks: List of [B, L, L] masks for each head
        """
        B, num_heads, num_blocks = scores.shape
        L = num_blocks * self.block_size
        
        block_masks = []
        
        for head in range(num_heads):
            head_scores = scores[:, head, :]  # [B, num_blocks]
            
            # Softmax normalization
            head_scores = F.softmax(head_scores, dim=-1)  # [B, num_blocks]
            
            # Cumulative sum for threshold selection
            sorted_scores, sorted_indices = torch.sort(head_scores, dim=-1, descending=True)
            cumulative_sum = torch.cumsum(sorted_scores, dim=-1)
            
            # Find cutoff point
            cutoff_indices = torch.sum(cumulative_sum <= threshold, dim=-1)  # [B]
            
            # Create block mask
            mask = torch.zeros(B, L, L, device=scores.device, dtype=torch.bool)
            
            for b in range(B):
                num_selected = min(cutoff_indices[b] + 1, num_blocks)
                selected_blocks = sorted_indices[b, :num_selected]
                
                for block_idx in selected_blocks:
                    start = block_idx * self.block_size
                    end = min((block_idx + 1) * self.block_size, L)
                    mask[b, :, start:end] = True
                    mask[b, start:end, :] = True
            
            block_masks.append(mask)
        
        return block_masks
    
    def dynamic_threshold_optimization(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Optimize threshold per head using dynamic programming.
        
        Args:
            scores: [B, num_heads, num_blocks]
            
        Returns:
            optimized_thresholds: [num_heads]
        """
        if not self.use_dynamic_threshold:
            return torch.ones(self.num_heads, device=scores.device) * self.threshold
        
        B, num_heads, num_blocks = scores.shape
        thresholds = torch.ones(num_heads, device=scores.device)
        
        # Simple heuristic: adjust based on score distribution
        for head in range(num_heads):
            head_scores = scores[:, head, :].flatten()
            
            # Compute statistics
            mean_score = head_scores.mean()
            std_score = head_scores.std()
            
            # Adjust threshold based on distribution
            if mean_score > 0.1:
                thresholds[head] = self.threshold * 0.9
            elif std_score > 0.05:
                thresholds[head] = self.threshold * 1.1
            else:
                thresholds[head] = self.threshold
        
        return torch.clamp(thresholds, 0.5, 0.95)
    
    def sparse_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        block_masks: List[torch.Tensor],
        causal: bool = False
    ) -> torch.Tensor:
        """
        Compute sparse attention using selected blocks.
        
        Args:
            Q: [B, num_heads, L, head_dim]
            K: [B, num_heads, L, head_dim]
            V: [B, num_heads, L, head_dim]
            block_masks: List of [B, L, L] masks for each head
            causal: whether to apply causal masking
            
        Returns:
            output: [B, num_heads, L, head_dim]
        """
        B, num_heads, L, head_dim = Q.shape
        output = torch.zeros_like(Q)
        
        for head in range(num_heads):
            mask = block_masks[head]  # [B, L, L]
            
            Q_head = Q[:, head, :, :]  # [B, L, head_dim]
            K_head = K[:, head, :, :]  # [B, L, head_dim]
            V_head = V[:, head, :, :]  # [B, L, head_dim]
            
            # Compute attention scores
            scores = torch.bmm(Q_head, K_head.transpose(-2, -1)) / self.scale  # [B, L, L]
            
            # Apply causal mask if needed
            if causal:
                causal_mask = torch.triu(torch.ones(L, L, device=Q.device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(causal_mask, float('-inf'))
            
            # Apply block mask
            scores = scores.masked_fill(~mask, float('-inf'))
            
            # Softmax and apply attention
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            out_head = torch.bmm(attn_weights, V_head)  # [B, L, head_dim]
            output[:, head, :, :] = out_head
        
        return output
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        XAttention forward pass with antidiagonal scoring and block selection.
        
        Args:
            query: [B, L, H]
            key: [B, L, H]
            value: [B, L, H]
            attention_mask: [B, L, L] or None
            causal: whether to apply causal masking
            
        Returns:
            output: [B, L, H]
            block_masks: List of selected block masks for each head
        """
        B, L, H = query.shape
        
        # Linear projections
        Q = self.q_proj(query)  # [B, L, H]
        K = self.k_proj(key)    # [B, L, H]
        V = self.v_proj(value)  # [B, L, H]
        
        # Reshape for multi-head attention
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        
        # Compute antidiagonal scores for block importance
        scores = self.compute_antidiagonal_scores(Q, K)  # [B, num_heads, num_blocks]
        
        # Dynamic threshold optimization
        thresholds = self.dynamic_threshold_optimization(scores)
        
        # Select important blocks
        block_masks = self.select_blocks(scores, thresholds[0])  # Use first threshold for all batches
        
        # Compute sparse attention
        sparse_out = self.sparse_attention(Q, K, V, block_masks, causal)  # [B, num_heads, L, head_dim]
        
        # Reshape and project output
        sparse_out = sparse_out.transpose(1, 2).contiguous().view(B, L, H)  # [B, L, H]
        output = self.out_proj(sparse_out)
        
        return output, block_masks
    
    def load_weights(self, state_dict: dict):
        """Load pre-trained weights if available."""
        self.load_state_dict(state_dict)
    
    def set_threshold(self, threshold: float):
        """Set global threshold for all heads."""
        self.threshold = threshold
        if hasattr(self, 'head_thresholds'):
            self.head_thresholds.fill_(threshold)
    
    def get_sparsity_stats(self, block_masks: List[torch.Tensor]) -> dict:
        """Compute sparsity statistics from block masks."""
        total_elements = 0
        selected_elements = 0
        
        for mask in block_masks:
            total_elements += mask.numel()
            selected_elements += mask.sum().item()
        
        sparsity = 1.0 - (selected_elements / total_elements)
        density = selected_elements / total_elements
        
        return {
            'sparsity': sparsity,
            'density': density,
            'selected_blocks': selected_elements,
            'total_blocks': total_elements
        }