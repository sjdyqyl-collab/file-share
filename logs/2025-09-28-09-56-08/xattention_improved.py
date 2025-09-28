import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class ThresholdPredictor(nn.Module):
    """
    Lightweight neural network for hierarchical threshold prediction.
    
    Replaces expensive dynamic programming with learned prediction.
    """
    
    def __init__(self, 
                 input_dim: int = 32,
                 hidden_dim: int = 64,
                 output_dim: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()  # Output threshold between 0 and 1
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict threshold based on head statistics.
        
        Args:
            x: Head statistics [B, input_dim]
            
        Returns:
            Predicted threshold [B, 1]
        """
        return self.mlp(x)

class SimpleKMeans:
    """Simple K-means clustering implementation without sklearn dependency."""
    
    @staticmethod
    def fit_predict(data: np.ndarray, n_clusters: int, max_iters: int = 100) -> np.ndarray:
        """
        Simple k-means clustering.
        
        Args:
            data: Input data [N, D]
            n_clusters: Number of clusters
            max_iters: Maximum iterations
            
        Returns:
            Cluster assignments [N]
        """
        N, D = data.shape
        
        # Initialize centroids randomly
        indices = np.random.choice(N, n_clusters, replace=False)
        centroids = data[indices]
        
        for _ in range(max_iters):
            # Assign points to nearest centroid
            distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([
                data[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centroids[k]
                for k in range(n_clusters)
            ])
            
            # Check convergence
            if np.all(centroids == new_centroids):
                break
            
            centroids = new_centroids
        
        return labels

class XAttentionImproved(nn.Module):
    """
    Improved XAttention with hierarchical threshold learning and adaptive features.
    
    Key improvements:
    1. Hierarchical threshold learning (replaces dynamic programming)
    2. Adaptive pattern detection
    3. Variable block sizes
    4. Memory-efficient pattern selection
    5. Cross-head correlation
    """
    
    def __init__(self, 
                 hidden_size: int,
                 num_heads: int,
                 block_sizes: list = [4, 8, 16],
                 stride: int = 8,
                 head_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 threshold_input_dim: int = 32):
        """
        Initialize improved XAttention.
        
        Args:
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
            block_sizes: List of block sizes for adaptive selection
            stride: Stride parameter for antidiagonal scoring
            head_dim: Dimension per head
            dropout: Dropout probability
            threshold_input_dim: Input dimension for threshold predictor
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.block_sizes = block_sizes
        self.stride = stride
        self.head_dim = head_dim or hidden_size // num_heads
        self.dropout = dropout
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Threshold predictor
        self.threshold_predictor = ThresholdPredictor(
            input_dim=threshold_input_dim,
            hidden_dim=64,
            output_dim=1
        )
        
        # Head clustering for cross-head correlation
        self.num_clusters = max(1, num_heads // 4)  # 4 heads per cluster
        self.register_buffer('head_clusters', torch.zeros(num_heads, dtype=torch.long))
        
        # Pattern templates
        self.pattern_types = ['antidiagonal', 'diagonal', 'vertical', 'horizontal']
        
    def compute_head_statistics(self, 
                              Q: torch.Tensor, 
                              K: torch.Tensor,
                              head_idx: int) -> torch.Tensor:
        """
        Compute statistics for threshold prediction.
        
        Args:
            Q: Queries [B, L, head_dim]
            K: Keys [B, L, head_dim]
            head_idx: Head index
            
        Returns:
            Statistics tensor [B, 32]
        """
        B, L, D = Q.shape
        
        # Compute attention statistics
        scale = 1.0 / np.sqrt(D)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, L, L]
        
        # Compute various statistics
        stats = []
        
        # 1. Attention entropy
        attn_probs = F.softmax(scores, dim=-1)
        entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8), dim=-1)  # [B, L]
        stats.append(entropy.mean(dim=-1, keepdim=True))  # [B, 1]
        stats.append(entropy.std(dim=-1, keepdim=True))   # [B, 1]
        
        # 2. Attention concentration
        max_attn = attn_probs.max(dim=-1)[0]  # [B, L]
        stats.append(max_attn.mean(dim=-1, keepdim=True))  # [B, 1]
        stats.append(max_attn.std(dim=-1, keepdim=True))   # [B, 1]
        
        # 3. Query/Key statistics
        q_norm = torch.norm(Q, dim=-1)  # [B, L]
        k_norm = torch.norm(K, dim=-1)  # [B, L]
        stats.append(q_norm.mean(dim=-1, keepdim=True))  # [B, 1]
        stats.append(q_norm.std(dim=-1, keepdim=True))   # [B, 1]
        stats.append(k_norm.mean(dim=-1, keepdim=True))  # [B, 1]
        stats.append(k_norm.std(dim=-1, keepdim=True))   # [B, 1]
        
        # 4. Sequence length scaling
        stats.append(torch.ones(B, 1, device=Q.device) * np.log(L))  # [B, 1]
        
        # 5. Head-specific features
        head_features = torch.zeros(B, 8, device=Q.device)
        head_features[:, head_idx % 8] = 1.0
        stats.append(head_features)
        
        # 6. Cross-head correlation features
        cluster_features = torch.zeros(B, 8, device=Q.device)
        cluster_id = self.head_clusters[head_idx].item()
        if cluster_id < 8:
            cluster_features[:, cluster_id] = 1.0
        stats.append(cluster_features)
        
        # Concatenate all statistics
        stats_tensor = torch.cat(stats, dim=-1)  # [B, 32]
        
        return stats_tensor
    
    def select_optimal_block_size(self, 
                                Q_block: torch.Tensor,
                                K_block: torch.Tensor) -> int:
        """
        Select optimal block size based on attention entropy.
        
        Args:
            Q_block: Query block [B, L, head_dim]
            K_block: Key block [B, L, head_dim]
            
        Returns:
            Selected block size
        """
        B, L, D = Q_block.shape
        
        # Compute attention entropy for different scales
        entropies = []
        for block_size in self.block_sizes:
            if L >= block_size:
                # Compute entropy for this block size
                num_blocks = L // block_size
                total_entropy = 0.0
                
                for b in range(num_blocks):
                    start = b * block_size
                    end = min((b + 1) * block_size, L)
                    
                    q_sub = Q_block[:, start:end, :]
                    k_sub = K_block[:, start:end, :]
                    
                    scale = 1.0 / np.sqrt(D)
                    scores = torch.matmul(q_sub, k_sub.transpose(-2, -1)) * scale
                    
                    attn_probs = F.softmax(scores, dim=-1)
                    entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8), dim=-1).mean()
                    total_entropy += entropy.item()
                
                entropies.append(total_entropy / num_blocks)
            else:
                entropies.append(float('inf'))
        
        # Select block size with lowest entropy (most focused attention)
        min_idx = np.argmin(entropies)
        return self.block_sizes[min_idx]
    
    def compute_multi_pattern_scores(self, 
                                   Q_block: torch.Tensor,
                                   K_block: torch.Tensor,
                                   pattern_type: str) -> torch.Tensor:
        """
        Compute scores using different pattern types.
        
        Args:
            Q_block: Query block [B, L, head_dim]
            K_block: Key block [B, L, head_dim]
            pattern_type: Type of pattern to use
            
        Returns:
            Pattern scores [B, L, L]
        """
        B, L, D = Q_block.shape
        
        if pattern_type == 'antidiagonal':
            return self._compute_antidiagonal_pattern(Q_block, K_block)
        elif pattern_type == 'diagonal':
            return self._compute_diagonal_pattern(Q_block, K_block)
        elif pattern_type == 'vertical':
            return self._compute_vertical_pattern(Q_block, K_block)
        elif pattern_type == 'horizontal':
            return self._compute_horizontal_pattern(Q_block, K_block)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    def _compute_antidiagonal_pattern(self, Q_block: torch.Tensor, K_block: torch.Tensor) -> torch.Tensor:
        """Compute antidiagonal pattern scores."""
        B, L, D = Q_block.shape
        
        # Reshape for antidiagonal computation
        Q_reshaped = self._reshape_antidiagonal(Q_block, self.stride)
        K_reshaped = self._reshape_antidiagonal(K_block, self.stride)
        
        scale = 1.0 / np.sqrt(D * self.stride)
        scores = torch.matmul(Q_reshaped, K_reshaped.transpose(-2, -1)) * scale
        
        return F.softmax(scores, dim=-1)
    
    def _compute_diagonal_pattern(self, Q_block: torch.Tensor, K_block: torch.Tensor) -> torch.Tensor:
        """Compute diagonal pattern scores."""
        B, L, D = Q_block.shape
        
        # Extract diagonal elements
        diagonal_mask = torch.eye(L, device=Q_block.device, dtype=torch.bool)
        scores = torch.matmul(Q_block, K_block.transpose(-2, -1))
        scores = scores.masked_fill(~diagonal_mask.unsqueeze(0), float('-inf'))
        
        return F.softmax(scores, dim=-1)
    
    def _compute_vertical_pattern(self, Q_block: torch.Tensor, K_block: torch.Tensor) -> torch.Tensor:
        """Compute vertical pattern scores."""
        B, L, D = Q_block.shape
        
        # Vertical attention pattern (column-wise)
        scores = torch.matmul(Q_block, K_block.transpose(-2, -1))
        
        # Emphasize vertical patterns
        vertical_weights = torch.ones(L, L, device=Q_block.device)
        for i in range(L):
            vertical_weights[i, :] = torch.exp(-torch.abs(torch.arange(L, device=Q_block.device) - i) / 10.0)
        
        scores = scores * vertical_weights.unsqueeze(0)
        return F.softmax(scores, dim=-1)
    
    def _compute_horizontal_pattern(self, Q_block: torch.Tensor, K_block: torch.Tensor) -> torch.Tensor:
        """Compute horizontal pattern scores."""
        B, L, D = Q_block.shape
        
        # Horizontal attention pattern (row-wise)
        scores = torch.matmul(Q_block, K_block.transpose(-2, -1))
        
        # Emphasize horizontal patterns
        horizontal_weights = torch.ones(L, L, device=Q_block.device)
        for j in range(L):
            horizontal_weights[:, j] = torch.exp(-torch.abs(torch.arange(L, device=Q_block.device) - j) / 10.0)
        
        scores = scores * horizontal_weights.unsqueeze(0)
        return F.softmax(scores, dim=-1)
    
    def _reshape_antidiagonal(self, x: torch.Tensor, stride: int) -> torch.Tensor:
        """Reshape tensor for antidiagonal computation."""
        B, L, D = x.shape
        reshaped = []
        
        for i in range(stride-1, -1, -1):
            reshaped.append(x[:, i::stride, :])
        
        return torch.cat(reshaped, dim=1)
    
    def memory_efficient_pattern_selection(self, 
                                         Q: torch.Tensor,
                                         K: torch.Tensor,
                                         threshold: float) -> torch.Tensor:
        """
        Memory-efficient pattern selection using streaming computation.
        
        Args:
            Q: Queries [B, L, head_dim]
            K: Keys [B, L, head_dim]
            threshold: Selection threshold
            
        Returns:
            Sparse mask [B, L, L]
        """
        B, L, D = Q.shape
        
        # Select optimal block size
        block_size = self.select_optimal_block_size(Q, K)
        
        # Initialize mask
        mask = torch.zeros(B, L, L, dtype=torch.bool, device=Q.device)
        
        # Process in streaming fashion
        num_blocks = L // block_size
        
        for b in range(num_blocks):
            start_idx = b * block_size
            end_idx = min((b + 1) * block_size, L)
            
            # Extract current block
            q_block = Q[:, start_idx:end_idx, :]
            k_block = K[:, start_idx:end_idx, :]
            
            # Select best pattern for this block
            best_pattern = None
            best_score = float('-inf')
            
            for pattern_type in self.pattern_types:
                pattern_scores = self.compute_multi_pattern_scores(q_block, k_block, pattern_type)
                score = pattern_scores.mean()
                
                if score > best_score:
                    best_score = score
                    best_pattern = pattern_type
            
            # Apply best pattern
            pattern_scores = self.compute_multi_pattern_scores(q_block, k_block, best_pattern)
            
            # Create block mask
            block_mask = torch.zeros(block_size, block_size, dtype=torch.bool, device=Q.device)
            
            # Select important positions
            flat_scores = pattern_scores.view(B, -1)
            for batch_idx in range(B):
                scores_batch = flat_scores[batch_idx]
                sorted_scores, sorted_indices = torch.sort(scores_batch, descending=True)
                
                cumulative_sum = torch.cumsum(sorted_scores, dim=0)
                total_sum = cumulative_sum[-1]
                target_sum = total_sum * threshold
                
                # Find minimal set
                selected_count = 0
                for i in range(len(sorted_scores)):
                    if cumulative_sum[i] >= target_sum:
                        selected_count = i + 1
                        break
                
                # Update mask
                selected_indices = sorted_indices[:selected_count]
                for idx in selected_indices:
                    row = idx // block_size
                    col = idx % block_size
                    if row < block_size and col < block_size:
                        block_mask[row, col] = True
            
            # Apply block mask to full mask
            mask[:, start_idx:end_idx, start_idx:end_idx] = block_mask.unsqueeze(0)
        
        return mask
    
    def update_head_clusters(self, Q: torch.Tensor, K: torch.Tensor):
        """Update head clusters based on attention pattern similarity."""
        B, H, L, D = Q.shape
        
        # Compute attention patterns for each head
        patterns = []
        for h in range(H):
            head_q = Q[:, h, :, :]
            head_k = K[:, h, :, :]
            
            scale = 1.0 / np.sqrt(D)
            scores = torch.matmul(head_q, head_k.transpose(-2, -1)) * scale
            patterns.append(scores.mean(dim=0))  # Average over batch
        
        patterns = torch.stack(patterns, dim=0)  # [H, L, L]
        
        # Flatten patterns for clustering
        flat_patterns = patterns.view(H, -1).cpu().numpy()  # [H, L*L]
        
        # Simple k-means clustering
        if self.num_clusters > 1:
            try:
                clusters = SimpleKMeans.fit_predict(flat_patterns, self.num_clusters)
            except:
                # Fallback to simple clustering
                clusters = np.random.randint(0, self.num_clusters, size=H)
        else:
            clusters = np.zeros(H, dtype=int)
        
        self.head_clusters = torch.tensor(clusters, device=Q.device)
        
        # Store cluster features for threshold prediction
        cluster_features = []
        for c in range(self.num_clusters):
            mask = (self.head_clusters.cpu().numpy() == c)
            if np.sum(mask) > 0:
                cluster_patterns = patterns[mask]  # [n_heads_in_cluster, L, L]
                cluster_feat = cluster_patterns.mean(dim=0)  # [L, L]
                cluster_features.append(cluster_feat.flatten()[:8])  # Take first 8 features
            else:
                cluster_features.append(torch.zeros(8))
        
        self.cluster_features = torch.stack([torch.tensor(cf) for cf in cluster_features], dim=0)  # [num_clusters, 8]
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                use_hierarchical_threshold: bool = True) -> torch.Tensor:
        """
        Forward pass of improved XAttention.
        
        Args:
            hidden_states: Input tensor [B, L, hidden_size]
            attention_mask: Optional attention mask
            use_hierarchical_threshold: Whether to use learned thresholds
            
        Returns:
            Output tensor [B, L, hidden_size]
        """
        B, L, _ = hidden_states.shape
        
        # Project to Q, K, V
        Q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Update head clusters if needed
        if not hasattr(self, 'cluster_features'):
            self.update_head_clusters(Q, K)
        
        # Initialize sparse mask
        sparse_mask = torch.zeros(B, self.num_heads, L, L, dtype=torch.bool, device=hidden_states.device)
        
        # Predict thresholds using hierarchical learning
        thresholds = torch.zeros(self.num_heads, device=hidden_states.device)
        
        for h in range(self.num_heads):
            # Get cluster for this head
            cluster_id = self.head_clusters[h]
            
            # Compute head statistics
            head_q = Q[:, h, :, :]  # [B, L, head_dim]
            head_k = K[:, h, :, :]  # [B, L, head_dim]
            
            stats = self.compute_head_statistics(head_q, head_k, h)
            
            # Average statistics across batch
            avg_stats = stats.mean(dim=0, keepdim=True)  # [1, 32]
            
            if use_hierarchical_threshold:
                # Use learned threshold predictor
                threshold = self.threshold_predictor(avg_stats).squeeze() * 0.9 + 0.1
            else:
                # Fallback to fixed threshold
                threshold = 0.8
            
            thresholds[h] = threshold
            
            # Memory-efficient pattern selection
            head_mask = self.memory_efficient_pattern_selection(head_q, head_k, threshold)
            sparse_mask[:, h, :, :] = head_mask
        
        # Apply attention mask if provided
        if attention_mask is not None:
            sparse_mask = sparse_mask & attention_mask.unsqueeze(1).unsqueeze(1)
        
        # Compute sparse attention
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~sparse_mask, float('-inf'))
        
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        
        output = torch.matmul(attn_probs, V)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        output = self.out_proj(output)
        
        return output
    
    def load_weights(self, state_dict: dict):
        """Load pre-trained weights."""
        self.load_state_dict(state_dict)
    
    def get_improvement_stats(self) -> dict:
        """Get statistics about improvements."""
        return {
            'num_clusters': self.num_clusters,
            'block_sizes': self.block_sizes,
            'pattern_types': self.pattern_types,
            'stride': self.stride
        }