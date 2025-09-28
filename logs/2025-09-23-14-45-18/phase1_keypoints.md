# Phase 1: Key Points Extraction

## Paper Title
Compact Attention: Exploiting Structured Spatio-Temporal Sparsity for Fast Video Generation

## Core Problem
The computational demands of self-attention mechanisms in transformer-based video generation, especially for ultra-long sequences, pose a critical challenge. Current approaches like factorized attention and fixed sparse patterns fail to fully exploit inherent spatio-temporal redundancies in video data.

## Key Insights
1. **Structured Sparsity Patterns**: Attention matrices exhibit structured yet heterogeneous sparsity patterns where specialized heads dynamically attend to distinct spatiotemporal regions (local, cross-shaped, or global patterns).

2. **Pattern Stability**: Attention patterns are stable across different inputs (prompts/seeds) and temporally robust within denoising steps, enabling offline pre-computation.

3. **3D Spatio-Temporal Redundancy**: Video data has unique 3D redundancies that existing sparse attention methods designed for language models fail to capture.

## Proposed Solution - Compact Attention
A hardware-aware acceleration framework with three innovations:
1. **Adaptive tiling strategies** that approximate diverse spatial interaction patterns via dynamic tile grouping
2. **Temporally varying windows** that adjust sparsity levels based on frame proximity
3. **Automated configuration search algorithm** that optimizes sparse patterns while preserving critical attention pathways

## Key Results
- Achieves 1.6-2.5× acceleration in attention computation on single-GPU setups
- Maintains comparable visual quality with full-attention baselines
- Tested on Wan2.1 (14B) and Hunyuan models
- Up to 62.36% sparsity with minimal quality degradation

## Technical Contributions
1. **Tile-based Deformable Sparse Pattern**: Hierarchical grouping mechanism respecting video's dual nature (temporal variation and spatial locality)
2. **Optimized Auto-Search**: Offline configuration pipeline with dual thresholds (recall τ and cost λ)
3. **Pattern Classification**: Systematic categorization of attention patterns (local, cross-shaped, global for spatial; time-variant/invariant for temporal)

## Experimental Validation
- Evaluated on 81-frame Wan2.1 and 129-frame Hunyuan at 768×1280 resolution
- Compared against STA, Sparse VideoGen, and SpargeAttn
- Metrics: SSIM, PSNR, MSE, VBench quality metrics, CLIPSIM, CLIP-T
- Achieved 2.51× speedup on Hunyuan with 62.36% sparsity while maintaining high quality