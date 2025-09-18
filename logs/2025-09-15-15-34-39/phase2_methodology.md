# Phase 2: Methodology Extraction - AdaSpa Paper

## Problem Formulation

### Blockified Sparse Attention Definition
Given:
- Sequence length L = f·h·w + t (video frames × height × width + text tokens)
- Block size B (default: 64)
- Number of blocks: L/B
- Block-level sparse pattern MS ∈ {0,1}^(L/B × L/B)
- Sparsity ratio: proportion of blocks to keep

### Mathematical Formulation
```
Wattn(gMS) = Softmax_safe(QK^T/√D - c(1-gMS))
```

Where:
- Q,K,V ∈ R^(H×L×D) (H: heads, L: sequence length, D: head dimension)
- gMS: expanded block mask from MS
- c: sufficiently large negative constant
- Wsum_attn ∈ R^(L/B × L/B): sum of attention weights within each block

### Optimal Sparse Indices
```
S* = argmax_S Wsum_attn(MS)
```
Select top-k blocks with highest Wsum_attn values, where k = (1-sparsity)×(L/B)²

## AdaSpa Architecture

### Two-Phase Search Strategy

#### Phase 1: Fused Online Search (Warmup Steps)
- **When**: Applied at warmup steps Ts = {10, 30} (default)
- **Process**: 
  1. First pass: Compute full FlashAttention and store LSE per row
  2. Second pass: Use cached LSE to compute Wsum_attn in block-wise manner
- **Output**: Block sparse mask for subsequent steps

#### Phase 2: LSE-Cached Online Search
- **When**: Applied between warmup steps
- **Process**: 
  1. Use cached LSE from previous Fused Online Search
  2. Single pass to compute Wsum_attn
  3. Generate new block sparse mask
- **Benefit**: Reduces search time by ~50%

### Head-adaptive Hierarchical Block Sparse Attention

#### Algorithm Steps
1. **Initial Assessment**: 
   - Compute Recall for each head at base sparsity
   - Sort heads by Recall performance
   - Count heads with Recall > 0.8 (threshold)

2. **Adaptive Adjustment**:
   - High-recall heads: increase sparsity to (1+sparsity)/2
   - Low-recall heads: decrease sparsity to (3×sparsity-1)/2
   - Maintain average sparsity across all heads

3. **Implementation**:
   - Apply different block masks per head
   - Ensure per-row uniform selection for continuity
   - Include text sink indices (video-text, text-video, text-text)

## Implementation Details

### Default Parameters
- **Sparsity**: 0.8 (80% sparse)
- **Block Size**: 64 tokens
- **Warmup Steps**: 10 steps full attention
- **Search Steps**: Ts = {10, 30}
- **Recall Threshold**: 0.8

### Optimization Techniques

#### Text Sink Enhancement
- Manually include all text-related attention indices
- Ensures text modality perception is preserved
- Indices: video-text, text-video, text-text interactions

#### Row-wise Uniform Selection
- Ensures each query attends to similar number of keys
- Prevents artifacts from "unimportant" regions
- Applied during block sparse pattern generation

### Memory and Computation
- **Memory Reduction**: O(L²) → O((1-sparsity)L²)
- **Search Overhead**: <5% of full attention time
- **Cache Storage**: LSE values from previous steps
- **Kernel Implementation**: Triton-based optimized kernels

## Integration Interface

### Usage Pattern
```python
from adaspa import adaspa_attention_handler

# Replace original attention with AdaSpa
q, k, v = get_qkv(hidden_states, qkv_weight)
out = adaspa_attention_handler(query=q, key=k, value=v)
```

### Compatibility
- Works with existing DiT architectures
- No model retraining required
- Compatible with FlashAttention 2
- Orthogonal to other acceleration techniques (parallelization, quantization, cache reuse)

## Complexity Analysis
- **Original**: O(L²d) time, O(L²) memory
- **AdaSpa**: O((1-sparsity)L²d) time, O((1-sparsity)L²) memory
- **Search Cost**: O(L²d) for Fused Online Search (amortized over many steps)
- **Storage**: O(L) for LSE cache, O((L/B)²) for block masks