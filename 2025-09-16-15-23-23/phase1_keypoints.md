# Phase 1: Key Points of AdaSpa Paper

## Problem Statement
- Long video generation with Diffusion Transformers (DiTs) is computationally expensive
- Attention mechanisms consume 83% of total FLOPs (500 PFLOPs out of 600 PFLOPs for 8-second 720p video)
- Existing sparse attention methods are inadequate for DiTs due to their unique characteristics

## Key Observations about DiT Attention Sparsity
1. **Hierarchical Structure**: Attention patterns are organized hierarchically between modalities (text vs video) and within video frames
2. **Blockified Patterns**: Due to hierarchical structure, continuous patterns (col, diag) fail; blockified patterns work better
3. **Dynamic Nature**: Sparse patterns vary significantly across inputs, layers, and attention heads
4. **Step Invariance**: While patterns change across inputs/layers/heads, they remain stable across denoising steps for a given layer/head
5. **LSE Stability**: Log-Sum-Exp distributions remain stable across denoising steps

## Proposed Solution: AdaSpa
- **First Dynamic Pattern + Online Precise Search method for DiTs**
- **Training-free and data-free** - no fine-tuning or dataset profiling required
- **Two key innovations**:
  1. Blockified pattern to capture hierarchical sparsity
  2. Fused LSE-Cached Search with Head-adaptive Hierarchical Block Sparse Attention

## Core Technical Contributions
1. **Blockified Sparse Attention**: Uses block-wise patterns instead of continuous patterns
2. **Fused LSE-Cached Online Search**: 
   - First pass: Computes full attention and caches LSE
   - Subsequent passes: Uses cached LSE for efficient precise search
3. **Head-adaptive Hierarchical Strategy**: Adjusts sparsity per attention head based on recall performance

## Performance Results
- **HunyuanVideo**: 1.78× speedup with 29.07 PSNR (vs 27.61 for Sparse VideoGen)
- **CogVideoX1.5**: 1.66× speedup with 23.25 PSNR (vs 18.98 for Sparse VideoGen)
- **Quality**: Maintains video quality with negligible degradation
- **Scalability**: Speedup increases with video length (up to 4.01× for 24s videos)