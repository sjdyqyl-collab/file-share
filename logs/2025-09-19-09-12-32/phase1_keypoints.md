# Phase 1: Key Points Extraction - DraftAttention Paper

## Title
DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance

## Problem Statement
- Video diffusion transformers (DiTs) suffer from high computational cost
- Attention mechanism accounts for >80% of total latency in video generation
- Generating 8 seconds of 720p video takes tens of minutes
- Quadratic complexity of attention becomes bottleneck with hundreds of thousands of tokens

## Key Innovation
DraftAttention: A training-free framework for accelerating video diffusion transformers using dynamic sparse attention on GPUs

## Core Technical Contributions

### 1. Low-Resolution Draft Attention
- Downsamples feature maps across frames using average pooling (8×16 kernel)
- Creates low-resolution draft attention map to identify important regions
- Reduces tokens by factor of 128 (from hundreds of thousands to manageable size)
- Exposes redundancy both spatially within frames and temporally across frames

### 2. Guided Sparse Attention
- Uses draft attention map to guide full-resolution sparse attention computation
- Reorders query, key, and value based on draft attention map
- Applies structured sparsity that aligns with hardware-optimized execution
- Restores original order after attention computation

### 3. Theoretical Foundation
- Provides bounds on approximation error between full and draft attention
- Shows error remains controlled and bounded
- Demonstrates low-resolution draft attention closely approximates full attention

### 4. Hardware-Friendly Design
- Deterministic reordering ensures contiguous memory access
- Aligns region-level sparsity with token-level computations
- Compatible with efficient attention frameworks (FlashAttention, Block Sparse Attention)

## Experimental Results
- Outperforms existing sparse attention methods in video generation quality
- Achieves up to 1.75× end-to-end speedup on GPUs
- Maintains generation quality while reducing computation
- Tested on HunyuanVideo and Wan2.1 models at 512p and 768p resolutions

## Key Advantages
1. **Efficiency**: Lightweight computation on reduced tokens
2. **Effectiveness**: Captures high-level representations and essential visual patterns
3. **Plug-and-Play**: No additional training required, seamless integration

## Technical Specifications
- Uses 8×16 pooling kernel with stride equal to kernel size
- Supports 128 visual tokens per kernel processing
- Retains full attention for first 25% of denoising steps
- Compatible with Block Sparse Attention implementation