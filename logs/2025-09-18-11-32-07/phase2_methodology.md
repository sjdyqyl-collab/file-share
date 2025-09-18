# Phase 2: Methodology of XAttention

## Overview
XAttention is a plug-and-play framework for accelerating long-context inference in Transformer models using block sparse attention with antidiagonal scoring. The method comprises three primary components: importance prediction, block selection, and threshold prediction.

## 1. Importance Prediction via Antidiagonal Scoring

### Core Insight
The sum of antidiagonal values (from lower-left to upper-right) in attention blocks serves as a powerful proxy for block importance.

### Technical Details
- **Block Size**: B×B attention blocks
- **Stride**: S parameter for strided sampling along antidiagonals
- **Scoring Process**:
  1. Within each B×B block, select elements along antidiagonal using stride S
  2. Compute sum of these selected elements as importance score
  3. Apply softmax normalization to create probability distribution

### Mathematical Formulation
For each block b, compute:
```
score(b) = Σ antidiagonal_elements(QK^T/√d_h)
```

### Advantages
1. **Information Preservation**: Each token contributes to at least one antidiagonal sum
2. **Pattern Detection**: Antidiagonal intersects both vertical and slash patterns within blocks
3. **Computational Efficiency**: Simple summation operation with minimal overhead

## 2. Block Selection Algorithm

### Process Flow
1. **Antidiagonal Summation**: Compute scores for each antidiagonal within blocks
2. **Softmax Normalization**: Convert scores to probability distribution
3. **Threshold-based Selection**: Select minimal set of blocks where cumulative probability ≥ τ

### Algorithm
```
Input: Query Q, Key K, block size B, stride S, threshold τ
Output: Sparse mask M

For each block b:
  1. Extract Q_slice = Q[bB:(b+1)B, :]
  2. Reshape Q and K along antidiagonals with stride S
  3. Compute approximate attention A_approx = softmax(Q_reshaped K_reshaped^T / √(d_h·S))
  4. Find blocks using find_blocks(A_approx, τ)
  5. Concatenate block masks
```

### Selection Criteria
```
find_blocks(A, τ) = argmin_B {|B| : Σ_b∈B Σ_(i,j)∈b A_i,j ≥ τ}
```

## 3. Dynamic Threshold Prediction

### Problem Formulation
- H attention heads with varying sparsity levels
- Dynamic programming to optimize thresholds per head

### Dynamic Programming Table
- D[h][m]: Best performance with m threshold adjustments across first h heads
- Recurrence: D[h][m] = max(D[h-1][m], P(h,m))

### Threshold Adjustment
- Start with τ = 0.9
- Reduce by 10% at each step: τ(m) = τ(m-1) × 0.9
- Maximum adjustments: M = 1000

## 4. Implementation Details

### Parameters
- **Stride S**: {4, 8, 16, 64} - Controls sparsity vs accuracy trade-off
- **Block Size B**: Typically 8×8 or 16×16
- **Threshold τ**: {0.9, 0.95} for different applications

### Warmup Strategy (for Video Generation)
- Use full attention for first 5 denoising steps
- Switch to XAttention for remaining steps
- Prevents layout shifts in generated videos

## 5. Computational Complexity

### Baseline (Full Attention)
- Time: [L, L, d] where L is sequence length, d is hidden dimension
- Complexity: O(L²d)

### XAttention
- **Pattern Selection**: O(L²/S) where S is stride
- **Sparse Attention**: O(L²·ρ) where ρ is density ratio (typically 0.07-0.55)
- **Total**: O(L²·(1/S + ρ))

### Speedup Factors
- Pattern selection: 24.9× faster than MInference
- Overall attention: Up to 13.5× speedup at 256k tokens
- Memory reduction: Proportional to sparsity ratio

## 6. Applications

### Natural Language Processing
- Models: Llama-3.1-8B-Instruct
- Tasks: RULER (synthetic), LongBench (real-world)
- Config: S=8,16 with dynamic threshold prediction

### Video Understanding
- Models: Qwen2-VL-7B-Instruct
- Tasks: VideoMME benchmark
- Config: S=16, τ=0.9

### Video Generation
- Models: HunyuanVideo (DiT architecture)
- Tasks: VBench prompts
- Config: S=8, τ={0.9,0.95}, 5-step warmup