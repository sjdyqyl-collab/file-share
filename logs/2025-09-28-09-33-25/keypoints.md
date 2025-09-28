# XAttention: Key Points Summary

## Abstract
XAttention is a plug-and-play framework that dramatically accelerates long-context inference in Transformer models using sparse attention. The key innovation is using antidiagonal sums as a proxy for block importance, achieving up to 13.5× acceleration while maintaining accuracy comparable to full attention.

## Main Problem
- Long-Context Transformer Models (LCTMs) suffer from quadratic computational complexity in attention mechanisms
- Block-sparse attention methods struggle to balance accuracy and efficiency due to costly block importance measurements
- Existing methods have high overhead for determining block importance, negating sparsity gains

## Proposed Solution: XAttention
A three-step framework:
1. **Strided Antidiagonal Scoring**: Score each block by summing values along strided antidiagonals
2. **Block Selection**: Select high-scoring blocks based on threshold
3. **Block Sparse Attention**: Compute attention only on selected blocks

## Key Innovations
- **Antidiagonal Scoring**: Uses sum of antidiagonal values as efficient proxy for block importance
- **Strided Pattern**: Intersects both vertical and slash patterns within blocks for robust detection
- **Dynamic Thresholding**: Uses dynamic programming to optimize thresholds per attention head
- **Plug-and-Play**: No retraining required, works with existing models

## Main Results
- **Accuracy**: Comparable to full attention across benchmarks (RULER, LongBench, VideoMME, VBench)
- **Speed**: Up to 13.5× acceleration in attention computation
- **Sparsity**: Achieves 6-55% density depending on context length
- **Robustness**: Maintains performance across 4k-256k token sequences
- **Applications**: Tested on Llama-3.1-8B, Qwen2-VL-7B, and HunyuanVideo models

## Technical Details
- Block size: 8×8 or 16×16
- Stride values: S=4, 8, 16, 64 (S=8, 16 recommended)
- Threshold: τ=0.9 default, dynamic optimization available
- Warmup: 5 steps of full attention for video generation