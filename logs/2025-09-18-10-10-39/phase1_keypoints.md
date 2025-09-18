# Phase 1: Key Points Extraction - XAttention

## Problem Statement
- Long-Context Transformer Models (LCTMs) suffer from high computational costs due to attention's quadratic complexity
- Block-sparse attention methods struggle to balance accuracy and efficiency due to costly block importance measurements
- Existing methods use computationally intensive solutions like token pooling which negate sparsity gains

## Key Innovation
- **Antidiagonal Scoring**: The sum of antidiagonal values (lower-left to upper-right) in attention matrices serves as a powerful proxy for block importance
- This enables precise identification and pruning of non-essential blocks
- Achieves high sparsity without sacrificing accuracy

## Three-Step Process
1. **Strided Antidiagonal Scoring**: Score blocks by summing values along strided antidiagonals
2. **Block Selection**: Select high-scoring blocks based on threshold
3. **Block Sparse Attention**: Compute attention only on selected blocks

## Main Contributions
- **Plug-and-play framework** that dramatically accelerates long-context inference
- **Training-free approach** - no retraining required
- **Up to 13.5× acceleration** in attention computation
- **Maintains accuracy comparable to full attention** across benchmarks
- **Validated on diverse domains**: language (RULER, LongBench), video understanding (VideoMME), video generation (VBench)

## Technical Advantages
- **Lightweight computation**: Antidiagonal scoring is computationally efficient
- **Robust pattern detection**: Antidiagonals intersect all vertical and slash patterns within blocks
- **Dynamic thresholding**: Uses dynamic programming to optimize thresholds per attention head
- **Warmup strategy**: For video generation, uses 5 steps of full attention before switching to sparse

## Performance Highlights
- **Language tasks**: Outperforms FlexPrefill and maintains accuracy at 128k tokens
- **Video understanding**: Outperforms full attention on long videos (1 hour)
- **Video generation**: Achieves PSNR 23.5, SSIM 0.822, LPIPS 0.155 with 45.5% density
- **Efficiency**: 24.9× faster pattern selection than MInference, 5.9× faster than FlexPrefill