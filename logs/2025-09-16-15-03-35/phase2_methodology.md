# Phase 2: Methodology of AdaSpa

## 4.1 Problem Formulation

### Blockified Sparse Attention Definition
- **Block size**: B (typically 64)
- **Sequence length**: L = f·h·w + t (video frames × height × width + text tokens)
- **Block-level pattern**: MS ∈ {0,1}^(L/B × L/B)
- **Sparse indices**: Set S of blocks to compute
- **Sparsity**: (1 - |S|/(L/B)²) × 100%

### Optimal Sparse Indices Selection
```
S* = argmax_S ||W_attn(g_MS)||
    = argmax_k∈{1,...,(1-sparsity)(L/B)²} Wsum_attn[k]
```

Where:
- Wsum_attn = Σ_i=0^B-1 Σ_j=0^B-1 W_attn[B·p+i, B·q+j]
- Complexity reduction: O(L²d) → O((1-sparsity)L²d)

## 4.2 Design of Adaptive Sparse Attention

### Two-Phase Search Strategy

#### Phase 1: Fused LSE-Cached Online Search
**Algorithm 1: Fused Online Search**
```
Input: Q, K, V
Output: LSE, Out, Wsum_attn

// First Pass: Full attention + LSE storage
1. Compute FlashAttention outputs
2. Store LSE for each row
3. Calculate Wsum_attn in block-wise manner

// Second Pass: Use cached LSE for Wsum_attn
1. Reuse LSE from first pass
2. Compute block importance scores
3. Select top-k blocks for sparse attention
```

#### Phase 2: LSE-Cached Online Search
**Algorithm 2: LSE-Cached Search**
```
Input: Q, K, LSE (cached)
Output: Wsum_attn

// Single pass using cached LSE
1. Use LSE from previous step
2. Compute block importance scores
3. Update sparse pattern if needed
```

### Head-Adaptive Hierarchical Block Sparse Attention

#### Adaptive Sparsity Mechanism
1. **Initial Setup**: Fixed sparsity (typically 0.8)
2. **Recall-based Adjustment**:
   - Sort heads by recall performance
   - n = number of heads with recall > 0.8
   - High-recall heads: increase sparsity to (1+sparsity)/2
   - Low-recall heads: decrease sparsity to (3×sparsity-1)/2
3. **Maintains average sparsity** while optimizing per-head performance

#### Hierarchical Structure
- **Global level**: Blockified pattern across entire sequence
- **Frame level**: Local patterns within each video frame
- **Head level**: Individual sparsity per attention head

### Implementation Details

#### Default Configuration
- **Sparsity**: 0.8 (80% sparsity)
- **Block size**: 64
- **Search steps**: Ts = {10, 30}
- **Warmup**: First 10 steps use full attention

#### Optimizations
1. **Text Sink**: Manually preserve video-text, text-video, text-text interactions
2. **Row-wise Uniformity**: Ensure each query attends to similar number of keys
3. **Kernel Implementation**: 2000+ lines Python, 1000+ lines Triton

#### Integration
```python
from adaspa import adaspa_attention_handler
# Single-line replacement
out = adaspa_attention_handler(query=q, key=k, value=v)
```

## 4.3 Computational Complexity Analysis

### Time Complexity
- **Full Attention**: O(L²d) where L = f·h·w + t
- **AdaSpa Sparse**: O((1-sparsity)L²d) = O(0.2L²d) with sparsity=0.8
- **Search Overhead**: <5% of full attention time due to LSE caching

### Memory Complexity
- **Full Attention**: O(L²) for attention matrix
- **AdaSpa**: O(L·B) for block-wise computation (B = block size)

### Runtime Representation
- **Baseline Full Attention**: Get_Time[L, d, L] = O(L²d)
- **AdaSpa**: Get_Time[0.2L, d, L] + Get_Time[L, d, 0.2L] = O(0.2L²d)
- **Search Cost**: Get_Time[L, d, B] + Get_Time[B, d, L] ≈ O(L·B·d)

## Key Technical Innovations

1. **Blockified Pattern**: Captures hierarchical sparsity structure
2. **LSE Caching**: Exploits step-invariance for efficient search
3. **Head Adaptation**: Optimizes sparsity per attention head
4. **Fused Search**: Combines full attention with pattern search
5. **Online Precise**: Real-time accurate sparse pattern identification