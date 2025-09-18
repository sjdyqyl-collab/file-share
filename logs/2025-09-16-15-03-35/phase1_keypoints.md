# Phase 1: Key Points of AdaSpa Paper

## Main Problem
- Diffusion Transformers (DiTs) suffer from high computational costs for long video generation
- Attention mechanisms consume ~83% of total FLOPs (500 PFLOPs out of 600 PFLOPs for 8-second 720p video)
- O(n²) complexity of attention becomes prohibitive for long sequences

## Key Observations about DiT Sparse Patterns
1. **Hierarchical and Blockified Structure**: 
   - Sparse patterns exhibit hierarchical structure between different modalities (video-video, video-text, text-text)
   - Within video modality, patterns are organized frame-by-frame
   - Continuous patterns (col, diag) fail due to hierarchical discontinuities
   - Blockified patterns achieve best recall (0.93-1.0 vs 0.12-0.54 for continuous patterns)

2. **Dynamic and Input-Dependent**:
   - Sparse patterns vary significantly across inputs, layers, and attention heads
   - Patterns remain invariant across denoising steps but change with prompts/seeds
   - Makes offline search unsuitable due to poor portability

3. **LSE Distribution Invariance**:
   - Log-Sum-Exp (LSE) distribution remains stable across denoising steps
   - Enables caching and reuse across steps

## Proposed Solution: AdaSpa
- **First method** to combine Dynamic Pattern + Online Precise Search
- **Training-free and data-free** - no fine-tuning or profiling required
- **Key innovations**:
  1. Blockified pattern to capture hierarchical sparsity
  2. Fused LSE-Cached Search with head-adaptive hierarchical block sparse attention
  3. Leverages step-invariance for efficient online search

## Performance Results
- **HunyuanVideo**: 1.78× speedup with PSNR=29.07 (vs 22.53 for MInference, 27.61 for Sparse VideoGen)
- **CogVideoX1.5-5B**: 1.66× speedup with PSNR=23.25 (vs 10.31 for MInference, 18.98 for Sparse VideoGen)
- **Quality preservation**: Maintains VBench scores of 80.13% and 81.90% respectively
- **Scalability**: Speedup increases to 4.01× for 24-second videos

## Technical Contributions
1. **Comprehensive sparsity analysis** revealing DiT-specific patterns
2. **Novel search strategy** reducing online search time to <5% of full attention time
3. **Head-adaptive mechanism** optimizing sparsity per attention head
4. **Plug-and-play implementation** with single-line integration

## Baseline Comparison
- **Full Attention**: Baseline with O(L²d) complexity
- **MInference**: Dynamic pattern with offline search + online approximation (1.27-1.39× speedup)
- **Sparse VideoGen**: Static pattern with online switching (1.52-1.58× speedup)
- **AdaSpa**: Dynamic pattern with online precise search (1.66-1.78× speedup)