# Phase 1: Key Points Extraction

## Original Abstract
Diffusion transformer–based video generation models (DiTs) have recently attracted widespread attention for their excellent generation quality. However, their computational cost remains a major bottleneck—attention alone accounts for over 80% of total latency, and generating just 8 seconds of 720p video takes tens of minutes—posing serious challenges to practical application and scalability. To address this, we propose the DraftAttention, a training-free framework for the acceleration of video diffusion transformers with dynamic sparse attention on GPUs. We apply down-sampling to each feature map across frames in the compressed latent space, enabling a higher-level receptive field over the latent composed of hundreds of thousands of tokens. The low-resolution draft attention map, derived from draft query and key, exposes redundancy both spatially within each feature map and temporally across frames. We reorder the query, key, and value based on the draft attention map to guide the sparse attention computation in full resolution, and subsequently restore their original order after the attention computation. This reordering enables structured sparsity that aligns with hardware-optimized execution. Our theoretical analysis demonstrates that the low-resolution draft attention closely approximates the full attention, providing reliable guidance for constructing accurate sparse attention. Experimental results show that our method outperforms existing sparse attention approaches in video generation quality and achieves up to 1.75× end-to-end speedup on GPUs.

## Key Points Summary

### Problem Statement
- Video diffusion transformers (DiTs) suffer from high computational costs
- Attention mechanism accounts for >80% of total latency
- Generating 8 seconds of 720p video takes tens of minutes
- Quadratic complexity with respect to context length becomes a bottleneck

### Proposed Solution: DraftAttention
- Training-free framework for accelerating video diffusion transformers
- Uses dynamic sparse attention on GPUs
- Two-stage approach:
  1. Compute low-resolution draft attention map using downsampled query/key
  2. Use draft map to guide sparse attention computation in full resolution

### Technical Innovation
- **Low-resolution draft attention**: Uses 8×16 pooling kernel with stride=kernel size, reducing tokens by 128×
- **Dynamic sparse patterns**: Generated per attention module (not static)
- **Hardware-friendly reordering**: Ensures contiguous memory layout for efficient execution
- **No training required**: Plug-and-play module for existing models

### Theoretical Contributions
- Error bounds for draft attention approximation
- Frobenius-norm bounds for sparsity mask error
- Proof that low-resolution draft attention closely approximates full attention

### Experimental Results
- **Models tested**: HunyuanVideo-T2V (768p, 128 frames), Wan2.1-T2V (512p/768p, 80 frames)
- **Performance**: Up to 1.75× end-to-end speedup on H100 GPU
- **Quality**: Outperforms Sparse VideoGen (SVG) in all metrics (PSNR, SSIM, LPIPS)
- **Sparsity ratios**: Tested 55%, 60%, 75%, 80%, 90%

### Key Advantages
1. **Efficiency**: Lightweight computation with reduced token count
2. **Effectiveness**: Preserves essential visual patterns
3. **Plug-and-play**: No additional training required
4. **Hardware optimization**: Structured sparsity for GPU efficiency