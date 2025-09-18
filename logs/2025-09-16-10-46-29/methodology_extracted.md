# Methodology Details - AdaSpa

## Problem Formulation

### Blockified Sparse Attention Definition
- **Input**: Sequence length L = f·h·w + t (frames × height × width + text tokens)
- **Block partitioning**: L divided into L/B chunks, B = block size (64)
- **Sparse pattern**: MS ∈ {0,1}^(L/B × L/B) with sparse indices S
- **Attention computation**: 
  ```
  Wattn(gMS) = Softmax_safe((QK^⊤)/√D - c(1-gMS))
  ```
  where c is sufficiently large negative bias

### Optimal Sparse Indices
- **Goal**: Maximize recall = Σ(selected attention weights) / Σ(all attention weights)
- **Approach**: 
  1. Compute Wsum_attn = sum of attention weights per block
  2. Select top-k blocks with highest Wsum_attn values
  3. Complexity reduction: O(L²d) → O((1-sparsity)L²d)

## Fused LSE-Cached Online Search

### Phase 1: Fused Online Search (Two-pass)
- **First Pass**: Compute full FlashAttention and store LSE (Log-Sum-Exp) per row
- **Second Pass**: Use cached LSE to compute Wsum_attn in block-wise manner
- **Algorithm**: 
  ```
  Pass 1: Compute attention outputs + store LSE
  Pass 2: Recompute attention weights using cached LSE to determine sparse blocks
  ```

### Phase 2: LSE-Cached Online Search (Single-pass)
- **Key insight**: LSE distribution remains stable across denoising steps
- **Process**: 
  1. Use LSE from previous search step
  2. Single pass to compute Wsum_attn
  3. 50% reduction in search time

## Head-adaptive Hierarchical Block Sparse Attention

### Strategy
1. **Initial assessment**: Compute recall for each head at fixed sparsity
2. **Adaptive adjustment**:
   - Heads with recall > 0.8: increase sparsity to (1+sparsity)/2
   - Heads with lowest recall: decrease sparsity to (3×sparsity-1)/2
3. **Benefits**: 
   - Maintains average sparsity
   - Improves accuracy for low-recall heads
   - Reduces redundancy in high-recall heads

## Implementation Details

### Configuration
- **Default sparsity**: 0.8
- **Block size**: 64
- **Search steps**: Ts = {10, 30}
- **Warmup**: 10 steps full attention

### Optimizations
1. **Text Sink**: Manually include all video-text, text-video, text-text interactions
2. **Row-wise uniform selection**: Ensures each query attends to similar number of keys
3. **Integration**: Single-line replacement for existing attention mechanisms

### System Architecture
- **Code base**: 2000+ lines Python, 1000+ lines Triton
- **Interface**: Plug-and-play adaspa_attention_handler
- **Compatibility**: Orthogonal to parallelization, quantization, cache reuse
- **Hardware**: Single A100 GPU-80GB for experiments