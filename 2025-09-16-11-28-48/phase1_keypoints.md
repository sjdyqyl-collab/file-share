# Phase 1: Key Points of AdaSpa Paper

## 1. Problem Statement
- **Challenge**: Generating high-fidelity long videos with Diffusion Transformers (DiTs) is computationally expensive due to attention mechanism complexity
- **Example**: 8-second 720p video (110K tokens) with HunyuanVideo requires ~600 PFLOPs, with ~500 PFLOPs consumed by attention

## 2. Key Observations
### Observation 1: Hierarchical Structure of Sparse Patterns
- DiTs exhibit hierarchical sparsity within and between different modalities (video-video, video-text, text-text)- Traditional continuous patterns (col, diag) fail due to fragmented structure
- Blockified patterns achieve better recall (0.93-1.0 vs 0.12-0.54 for continuous patterns)

### Observation 2: Dynamic Nature of Sparse Patterns
- Sparse patterns vary significantly across:
  - Different attention heads
  - Different layers
  - Different input prompts/seeds
- Offline search methods lack portability and accuracy

### Observation 3: Invariance Across Denoising Steps
- Sparse patterns and LSE distributions remain stable across diffusion denoising steps
- This enables caching and reuse strategies

## 3. Proposed Solution: AdaSpa
- **First Dynamic Pattern + Online Precise Search method for DiTs**
- **Training-free and data-free** - no additional fine-tuning or dataset profiling required
- **Two key innovations**:
  1. **Blockified Pattern**: Captures hierarchical sparsity efficiently
  2. **Fused LSE-Cached Search**: Enables precise online search with minimal overhead

## 4. Technical Contributions
- **Comprehensive Analysis**: First in-depth analysis of DiT attention sparsity characteristics
- **Novel Framework**: First effective combination of dynamic patterns and precise online search
- **Practical Implementation**: Plug-and-play solution with seamless integration

## 5. Performance Results
- **HunyuanVideo**: 1.78× speedup with PSNR 29.07 (vs 22.53 for MInference, 27.61 for Sparse VideoGen)
- **CogVideoX1.5-5B**: 1.66× speedup with PSNR 23.25 (vs 10.31 for MInference, 18.98 for Sparse VideoGen)
- **Scaling**: Achieves up to 4.01× speedup for 24-second videos

## 6. Key Advantages
- **Quality Preservation**: Maintains high video quality while achieving significant speedup
- **Adaptability**: Head-adaptive hierarchical block sparse attention
- **Efficiency**: Online search time <5% of full attention generation time
- **Compatibility**: Orthogonal to other acceleration techniques (parallelization, quantization, cache reuse)