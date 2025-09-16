# Phase 2: Methodology of AdaSpa

## 1. Problem Formulation

### Blockified Sparse Attention Definition
- **Block Size**: Partition sequence length L into L/B chunks of size B
- **Block-level Pattern**: MS ∈ {0,1}^(L/B × L/B) where S is set of sparse indices
- **Expansion**: Expand MS to gMS ∈ {0,1}^(L×L) with large negative bias -c(1-gMS)
- **Complexity Reduction**: From O(L²d) to O((1-sparsity)L²d)

### Optimal Sparse Indices
- **Goal**: Maximize recall of attention weights
- **Wsum_attn**: Sum of attention weights within each block
- **Optimal Selection**: S* = argmax Wsum_attn[k] for top-k blocks

## 2. Two-Phase Search Strategy

### Phase 1: Fused LSE-Cached Online Search

#### Fused Online Search (Two-Pass)
1. **First Pass**: Compute full FlashAttention and store LSE for each row
2. **Second Pass**: Use cached LSE to compute Wsum_attn in block-wise manner

#### LSE-Cached Online Search (One-Pass)
- **Leverage**: LSE similarity across denoising steps
- **Process**: Use cached LSE from previous search to compute new Wsum_attn
- **Benefit**: Reduces search time by half

### Phase 2: Head-adaptive Hierarchical Block Sparse Attention

#### Motivation
- Different heads exhibit varying sparsity characteristics
- Uniform sparsity across heads is suboptimal
- Individual head sparsity causes kernel load imbalance

#### Hierarchical Strategy
1. **Initial Setup**: Fixed sparsity for all heads
2. **Evaluation**: Compute recall for each head
3. **Adaptation**: 
   - Increase sparsity for heads with recall > 0.8: sparsity_new = (1+sparsity)/2
   - Decrease sparsity for heads with lowest recall: sparsity_new = (3×sparsity-1)/2
4. **Balance**: Maintain average sparsity while improving accuracy

## 3. Implementation Details

### Default Configuration
- **Sparsity**: 0.8
- **Block Size**: 64
- **Search Steps**: Ts = {10, 30}
- **Warmup**: First 10 steps use full attention

### Technical Implementation
- **Codebase**: 2000+ lines Python, 1000+ lines Triton
- **Integration**: One-line change with adaspa_attention_handler
- **Base**: Block-Sparse-Attention with FlashAttention 2

### Optimization Techniques
1. **Text Sink**: Manually preserve video-text, text-video, text-text interactions
2. **Row-wise Uniformity**: Ensure each query attends to similar number of keys

## 4. Algorithm Overview

### Algorithm 1: Fused Online Search
```
Input: Q, K, V
Output: LSE, Out, Wsum_attn

// First Pass: Compute FlashAttention and store LSE
1. Initialize lse, row_max, acc
2. For each key block k, value block v:
   - Compute qk = Dot(q, k)
   - Update row_max and compute softmax
   - Accumulate LSE and output
3. Store final LSE and output

// Second Pass: Compute Wsum_attn
4. For each key block k:
   - Use cached LSE to compute attention weights
   - Sum weights for each block position
```

### Algorithm 2: LSE-Cached Online Search
```
Input: Q, K, LSE (cached)
Output: Wsum_attn

// Single Pass: Use cached LSE
1. For each key block k:
   - Compute qk = Dot(q, k)
   - Use cached LSE for softmax computation
   - Sum weights for block positions
```

## 5. System Architecture

### Workflow
1. **Warmup Phase**: Steps 1 to tw-1 use full attention
2. **Initial Search**: Step tw performs Fused Online Search
3. **Cached Search**: Subsequent search steps use LSE-Cached approach
4. **Adaptive Application**: Head-adaptive sparsity applied to intermediate steps

### Integration
- **Plug-and-play**: Replace original attention with adaspa_attention_handler
- **Compatibility**: Works with existing DiTs without modification
- **Orthogonality**: Complementary to other acceleration methods