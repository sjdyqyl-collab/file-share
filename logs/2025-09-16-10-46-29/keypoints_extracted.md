# Key Points Extracted from AdaSpa Paper

## Abstract Summary
- **Problem**: Diffusion Transformers (DiTs) suffer from high computational costs in attention mechanisms for long video generation
- **Solution**: AdaSpa - first Dynamic Pattern + Online Precise Search sparse attention method
- **Key Features**: 
  - Blockified pattern for hierarchical sparsity in DiTs
  - Fused LSE-Cached Search with Head-adaptive Hierarchical Block Sparse Attention
  - Training-free, data-free, plug-and-play solution
- **Results**: Substantial acceleration across models while preserving video quality

## Core Observations
1. **Hierarchical Structure**: DiTs exhibit hierarchical sparsity between and within modalities
2. **Dynamic Patterns**: Sparse patterns vary with inputs, layers, and heads but remain invariant across denoising steps
3. **Blockified Structure**: Continuous patterns fail due to hierarchical discontinuities; blockified patterns achieve best recall

## Methodology Components
1. **Blockified Sparse Attention**: Uses block-wise attention with configurable sparsity
2. **Fused LSE-Cached Online Search**: Two-phase approach leveraging LSE invariance across steps
3. **Head-adaptive Hierarchical Strategy**: Adjusts sparsity per head based on recall performance

## Experimental Results
- **HunyuanVideo**: 1.78× speedup with maintained quality
- **CogVideoX1.5-5B**: 1.66× speedup with best quality metrics
- **Scaling**: Up to 4.01× speedup for 24-second videos
- **Comparison**: Outperforms Sparse VideoGen (1.58×) and MInference (1.27×)

## Technical Details
- **Sparsity**: Default 0.8
- **Block Size**: 64
- **Search Steps**: Ts={10,30}
- **Warmup**: 10 steps full attention
- **Implementation**: 2000+ lines Python, 1000+ lines Triton