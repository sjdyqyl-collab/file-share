# Phase 2: Detailed Methodology of AdaSpa

## 4.1 Problem Formulation: Blockified Sparse Attention

### Mathematical Formulation
- **Sequence Length**: L = f·h·w + t (frames × height × width + text tokens)
- **Block Size**: B (typically 64)
- **Block Pattern**: MS ∈ {0,1}^(L/B × L/B) where MS_ij = 1 indicates active block
- **Optimization Goal**: Maximize recall under sparsity constraint

### Optimal Sparse Indices Selection
```
S* = argmax_{S} ||W_sum_attn(MS)||
W_sum_attn[p,q] = Σ_{i=0}^{B-1} Σ_{j=0}^{B-1} W_attn[B·p+i, B·q+j]
```

## 4.2 Design Components

### 1. Fused LSE-Cached Online Search

#### Phase 1: Fused Online Search (Warmup)
- **Timing**: Performed at warmup steps (e.g., steps 10, 30)
- **Two-Pass Process**:
  1. **First Pass**: Full FlashAttention computation + LSE caching
  2. **Second Pass**: Block-wise W_sum_attn computation using cached LSE

#### Phase 2: LSE-Cached Online Search
- **Timing**: All subsequent steps
- **Single-Pass Process**: Uses cached LSE from Phase 1 for W_sum_attn computation
- **Time Reduction**: 50% reduction by eliminating first pass

### 2. Head-adaptive Hierarchical Block Sparse Attention

#### Adaptive Sparsity Strategy
1. **Initial Setup**: Fixed sparsity (e.g., 0.8) for all heads
2. **Performance Evaluation**: Compute recall for each head
3. **Hierarchical Adjustment**:
   - High recall heads (>0.8): Increase sparsity to (1+sparsity)/2
   - Low recall heads: Decrease sparsity to (3×sparsity-1)/2
4. **Balance**: Maintains average sparsity while optimizing per-head performance

### 3. Implementation Optimizations

#### Text Sink Enhancement
- **Manual Selection**: All video-text, text-video, text-text attention indices
- **Purpose**: Enhances text-video cross-modal perception

#### Row-wise Uniform Selection
- **Constraint**: Each query attends to roughly equal number of keys
- **Benefit**: Prevents artifacts from "unimportant" regions being ignored

## Algorithm Pseudocode

### Algorithm 1: Fused Online Search
```
Input: Q, K, V
Output: LSE, Out, W_sum_attn

// First Pass: Full FlashAttention + LSE caching
lse ← -∞, row_max ← 1, acc ← 0
for each key block k ∈ K, value block v ∈ V:
    qk ← Dot(q, k)
    row_max ← update(row_max, qk)
    p ← online_softmax(row_max, qk)
    lse += Sum(p, -1)
    acc ← Dot(p, v, acc)
LSE ← Log(lse) + row_max
Out ← acc

// Second Pass: W_sum_attn computation
for each key block k ∈ K:
    qk ← Dot(q, k)
    p ← Log(qk - LSE)
    p_sum = Sum(p)
    Store p_sum to W_sum_attn
```

### Algorithm 2: LSE-Cached Online Search
```
Input: Q, K, LSE (cached)
Output: W_sum_attn

// Single pass using cached LSE
for each key block k ∈ K:
    qk ← Dot(q, k)
    p ← Log(qk - LSE)
    p_sum = Sum(p)
    Store p_sum to W_sum_attn
```

## 4.3 Implementation Details

### Configuration Parameters
- **Sparsity**: 0.8 (default)
- **Block Size**: 64
- **Search Steps**: Ts = {10, 30}
- **Warmup Steps**: First 10 steps use full attention

### Integration
- **Plug-and-play**: Single line change from original attention
- **Code Base**: 2000+ lines Python, 1000+ lines Triton
- **Compatibility**: Works with FlashAttention 2 and Block-Sparse-Attention

### Runtime Complexity
- **Full Attention**: O(L²d)
- **Block Sparse Attention**: O((1-sparsity)L²d)
- **Search Overhead**: <5% of full attention time using optimized kernels