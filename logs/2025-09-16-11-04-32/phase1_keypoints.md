# Phase 1: Key Points Extraction - AdaSpa Paper

## Problem Statement
- Generating high-fidelity long videos with Diffusion Transformers (DiTs) is computationally expensive
- Attention mechanism is the major bottleneck: 8-second 720p video (110K tokens) with HunyuanVideo takes ~600 PFLOPs, with ~500 PFLOPs consumed by attention
- Attention complexity is O(n²) where n is sequence length

## Key Observations about DiT Sparse Patterns
1. **Hierarchical Structure**: DiTs exhibit hierarchical sparsity between and within different modalities (video-video, video-text, text-video, text-text)
2. **Blockified Pattern**: Due to hierarchical structure, continuous patterns (col, diag) fail; blockified patterns achieve best recall
3. **Dynamic Nature**: Sparse patterns vary with inputs, layers, and heads, making offline search unsuitable
4. **Invariance Property**: Sparse patterns and LSE (Log-Sum-Exp) remain invariant across denoising steps, enabling caching

## Proposed Solution - AdaSpa
- **First Dynamic Pattern + Online Precise Search method** for DiTs
- **Training-free and data-free**: No fine-tuning or dataset-dependent profiling required
- **Two key innovations**:
  1. Blockified pattern to capture hierarchical sparsity
  2. Fused LSE-Cached Search with Head-adaptive Hierarchical Block Sparse Attention

## Technical Contributions
1. **Comprehensive Analysis**: First in-depth analysis of sparse characteristics in DiT attention mechanisms
2. **Novel Framework**: First effective combination of Dynamic Pattern and Online Precise Search
3. **Efficient Implementation**: Plug-and-play solution with minimal overhead (<5% of full attention time)

## Performance Results
- **HunyuanVideo**: 1.78× speedup with 29.07 PSNR (vs 27.61 for Sparse VideoGen, 22.53 for MInference)
- **CogVideoX1.5-5B**: 1.66× speedup with 23.25 PSNR (vs 18.98 for Sparse VideoGen, 10.31 for MInference)
- **Scaling**: Up to 4.01× speedup for 24-second videos
- **Quality**: Maintains high fidelity with negligible quality loss

## Key Advantages
- **Adaptive**: Head-adaptive sparsity for different attention heads
- **Efficient**: Real-time precise search with minimal overhead
- **Scalable**: Speedup increases with video length
- **Practical**: One-line code change integration