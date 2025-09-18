# Phase 2: Methodology Extraction - XAttention

## Overview
XAttention is a three-component framework for efficient block-sparse attention in long-context transformers:
1. Importance prediction using antidiagonal scoring
2. Block selection based on threshold mechanism
3. Dynamic threshold prediction per attention head

## 1. Importance Prediction (Antidiagonal Scoring)

### Core Concept
- **Insight**: Sum of antidiagonal values in attention blocks serves as proxy for block importance
- **Pattern**: Strided antidiagonal selection captures both vertical and slash attention patterns
- **Efficiency**: Simple summation avoids complex pooling or search algorithms

### Technical Details
- **Block Size**: B×B attention blocks (typically 8×8 or 16×16)
- **Stride**: S parameter controls sampling density (S=8, 16, 32, 64)
- **Scoring**: For each block, sum values along strided antidiagonals
- **Pattern Coverage**: Antidiagonal intersects all possible vertical and slash patterns within a block

### Mathematical Formulation
For block b with attention values A:
```
score(b) = Σ antidiagonal_values(A, stride=S)
```

## 2. Block Selection Algorithm

### Process Flow
1. **Antidiagonal Summation**: Compute strided antidiagonal sums for each block
2. **Softmax Normalization**: Convert sums to probability distribution
3. **Threshold Selection**: Select minimal set of blocks exceeding cumulative threshold τ

### Algorithm Details
```python
# Pseudo-code for block selection
for each block b:
    Q_slice = Q[b*B:(b+1)*B, :]
    K_slice = K[b*B:(b+1)*B, :]
    
    # Reshape for antidiagonal computation
    Q_reshaped = reshape_antidiagonal(Q_slice, stride=S)
    K_reshaped = reshape_antidiagonal(K_slice, stride=S)
    
    # Approximate attention scores
    A_approx = softmax(Q_reshaped @ K_reshaped.T / sqrt(dh*S))
    
    # Select blocks based on threshold
    selected_blocks = find_blocks(A_approx, threshold=τ)
```

### Threshold Selection
- **Dynamic**: find_blocks() identifies minimal set for cumulative probability ≥ τ
- **Adaptive**: Threshold τ adjusts based on content and head characteristics
- **Efficiency**: Greedy selection minimizing number of blocks

## 3. Minimum Threshold Prediction

### Dynamic Programming Approach
- **Problem**: Optimize threshold per attention head for accuracy-efficiency trade-off
- **Formulation**: D[h][m] = best performance with m adjustments across first h heads
- **Recurrence**: D[h][m] = max(D[h-1][m], P(h, m))

### Implementation Details
- **Initial Threshold**: τ_start = 0.9
- **Reduction Factor**: 10% per adjustment step
- **Maximum Adjustments**: M = 1000
- **Average Resulting Threshold**: ~0.8

### Threshold Adjustment
```
th(m) = th(m-1) × 0.9
```

## 4. Computational Complexity Analysis

### Baseline (Full Attention)
- **Time**: O(L²d) where L=sequence length, d=head dimension
- **Memory**: O(L²) for attention matrix

### XAttention (Sparse)
- **Pattern Selection**: O(L²/S² × B²) for antidiagonal scoring
- **Sparse Attention**: O(L² × density) where density ∈ [6%, 55%]
- **Total Speedup**: Up to 13.5× for L=256K

### Matrix Multiplication Representation
- **Baseline**: [L, d, L] → L²d operations
- **XAttention**: [L, d, L×density] → L²d×density operations
- **Pattern Selection**: [L/S, L/S, B²/S²] for scoring

## 5. Implementation Details

### Parameters
- **Stride S**: Controls sparsity-accuracy trade-off (S=8, 16, 32)
- **Block Size B**: Typically 8 or 16
- **Threshold τ**: 0.9 default, optimized per-head via DP
- **Warmup Steps**: 5 steps for video generation (diffusion models)

### Compatibility
- **No Retraining**: Plug-and-play with existing models
- **Architecture Agnostic**: Works with causal and non-causal attention
- **Hardware Efficient**: Optimized for GPU execution

## 6. Special Considerations

### Video Generation Adaptation
- **Non-causal Attention**: XAttention adapted for diffusion transformers
- **Warmup Strategy**: 5 steps of full attention before switching to sparse
- **Quality Preservation**: PSNR > 23.5, SSIM > 0.82, LPIPS < 0.155

### Multimodal Applications
- **Video Understanding**: Qwen2-VL-7B with frame-wise processing
- **Cross-modal Attention**: Handles text-video attention patterns
- **Temporal Patterns**: Captures long-range temporal dependencies

## 7. Limitations and Assumptions

### Current Constraints
- **Block Granularity**: Fixed block sizes may miss fine-grained patterns
- **Stride Selection**: Manual tuning required for optimal S
- **Head Homogeneity**: Assumes similar sparsity patterns across heads (addressed by per-head thresholds)
- **Pattern Limitations**: Focuses on vertical/slash patterns, may miss others

### Assumptions
- **Attention Sparsity**: Assumes inherent sparsity in attention matrices
- **Pattern Stability**: Assumes attention patterns are stable within blocks
- **Threshold Generalization**: Assumes DP-optimized thresholds generalize across inputs