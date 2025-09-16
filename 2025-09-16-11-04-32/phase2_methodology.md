# Phase 2: Methodology Extraction - AdaSpa

## 4.1 Problem Formulation

### Blockified Sparse Attention Definition
- Partition length dimension L into L/B chunks, where B is block size
- Define block-level sparse pattern MS ∈ {0,1}^(L/B × L/B)
- Expand MS to gMS ∈ {0,1}^(L×L) and apply large negative bias -c(1-gMS)
- Attention computation: Wattn(gMS) = Softmax_safe(QK^T/√D - c(1-gMS))

### Optimal Sparse Indices
- Goal: Maximize recall by retaining maximum attention weights
- Define Wsum_attn as sum of attention weights within each block
- Optimal indices: S* = argmax_k Wsum_attn[k] for top-k blocks
- Complexity reduction: O(L²d) → O((1-sparsity)L²d)

## 4.2 Design of Adaptive Sparse Attention

### Fused LSE-Cached Online Search
**Two-phase approach:**

1. **Fused Online Search (Two-pass)**
   - First pass: Compute FlashAttention outputs and store LSE for each row
   - Second pass: Use cached LSE to compute Wsum_attn in block-wise manner
   - Algorithm 1 details the implementation

2. **LSE-Cached Online Search (One-pass)**
   - Leverage LSE similarity across denoising steps
   - Use cached LSE from previous search to calculate Wsum_attn
   - Reduces search time by half
   - Algorithm 2 details the implementation

### Head-adaptive Hierarchical Block Sparse Attention
**Hierarchical strategy:**
- Fix initial sparsity and compute recall for each head
- Sort heads by recall performance
- For n heads with recall > 0.8: increase sparsity to (1+sparsity)/2
- For n heads with lowest recall: decrease sparsity to (3×sparsity-1)/2
- Maintains average sparsity while optimizing per-head performance

## Implementation Details

### Configuration Parameters
- Default sparsity: 0.8
- Block size: 64
- Search steps: Ts = {10, 30}
- Warmup steps: First 10 steps use full attention

### Optimization Techniques
1. **Text Sink**: Manually select video-text, text-video, text-text indices
2. **Row Wise**: Ensure per-row uniform selection for video continuity

### Algorithm 1: Fused Online Search
```
Input: Q, K, V
Output: LSE, Out, Wsum_attn

// First Pass: Compute FlashAttention and store LSE
for each key block k∈K, value block v∈V:
    qk ← Dot(q, k)
    row_max ← update(row_max, qk)
    p ← online_softmax(row_max, qk)
    lse += Sum(p, -1)
    acc ← Dot(p, v, acc)
LSE ← Log(lse) + row_max
Out ← acc

// Second Pass: Compute Wsum_attn using cached LSE
for each key block k∈K:
    qk ← Dot(q, k)
    p ← Log(qk - LSE)
    p_sum = Sum(p)
    Store p_sum to Wsum_attn
```

### Algorithm 2: LSE-Cached Online Search
```
Input: Q, K, LSE
Output: Wsum_attn

// Single pass using cached LSE
for each key block k∈K:
    qk ← Dot(q, k)
    p ← Log(qk - LSE)
    p_sum = Sum(p)
    Store p_sum to Wsum_attn
```

## Architecture Overview
- **Warmup Phase**: Steps 1 to tw-1 use full attention
- **Search Phase**: At step tw, perform Fused Online Search
- **Cache Phase**: Subsequent steps use cached LSE for LSE-Cached Search
- **Head-adaptive**: Apply hierarchical sparsity adjustment per head

## Technical Specifications
- Implementation: 2,000+ lines Python, 1,000+ lines Triton code
- Integration: One-line code change via adaspa_attention_handler
- Compatibility: Works with FlashAttention 2
- Memory: Block-wise processing reduces memory footprint to O(Lb)