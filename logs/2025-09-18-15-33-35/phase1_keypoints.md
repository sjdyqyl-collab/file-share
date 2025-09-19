# Phase 1: Key Points Extraction - Compact Attention

## Problem Statement
- **Challenge**: Quadratic complexity of self-attention in transformer-based video generation models
- **Scale**: Generating 128-frame 720p HD video requires processing over 100K tokens
- **Impact**: Attention computation consumes 68-72% of total generation time in Hunyuan-video architecture

## Key Insight
- Attention matrices exhibit structured yet heterogeneous sparsity patterns
- Specialized heads dynamically attend to distinct spatiotemporal regions:
  - Local pattern (fine-grained detail synthesis)
  - Cross-shaped pattern (directional sensitivity)
  - Global pattern (full spatial connectivity)
  - Time-variant patterns (temporal relative distance correlation)
  - Time-invariant patterns (frame-agnostic distributions)

## Main Contributions
1. **Pattern Discovery**: Revealed structured and hierarchical attention patterns in video diffusion transformers
2. **Compact Attention Framework**: Training-free sparse attention framework with offline configuration search
3. **Performance**: Achieved 1.6-2.5× acceleration on single-GPU setups while maintaining visual quality

## Technical Innovations
1. **Adaptive tiling strategies**: Approximate diverse spatial interaction patterns via dynamic tile grouping
2. **Temporally varying windows**: Adjust sparsity levels based on frame proximity
3. **Automated configuration search**: Optimizes sparse patterns while preserving critical attention pathways

## Key Findings
- Attention patterns are stable across different inputs (prompts, seeds) and denoising steps
- Enables offline precomputation of attention masks
- Tile-based sparsity reduces computational overhead compared to token-level sparsity
- Early denoising steps are more sensitive to sparsification than later steps

## Experimental Results
- **Wan2.1 (80K tokens)**: 33.99% sparsity with 1.65× speedup
- **Hunyuan (127K tokens)**: 62.36% sparsity with 2.51× speedup
- Quality metrics (SSIM, PSNR, VBench scores) maintained comparable to full attention baselines