# Phase 1: Key Points of DraftAttention Paper

## Title
DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance

## Main Problem
- Video diffusion transformers (DiTs) suffer from extreme computational cost
- Attention mechanism accounts for >80% of total latency
- 8 seconds of 720p video generation takes tens of minutes
- Quadratic complexity with respect to context length becomes bottleneck

## Proposed Solution
- **DraftAttention**: Training-free framework for accelerating video diffusion transformers
- Uses dynamic sparse attention on GPUs with low-resolution guidance
- Two-stage approach:
  1. Compute low-resolution draft attention map via down-sampling
  2. Use draft map to guide sparse attention computation at full resolution

## Key Innovations
1. **Low-resolution draft attention**: Uses 8×16 pooling kernel (128× token reduction) to identify critical attention regions
2. **Dynamic sparsity**: Adapts sparse patterns per attention module (not static like prior work)
3. **Hardware-friendly reordering**: Reorders tokens to enable contiguous memory access for efficient sparse computation
4. **Training-free**: Works as plug-and-play module without retraining

## Theoretical Guarantees
- Bounded error between full and draft attention (Frobenius norm analysis)
- Error introduced by sparsity mask remains controlled
- Proved approximation quality for the two-stage approach

## Experimental Results
- **Speedup**: Up to 1.75× end-to-end acceleration on H100 GPUs
- **Quality**: Outperforms Sparse VideoGen (SVG) at same sparsity levels
- **Metrics**: Better PSNR, SSIM, LPIPS scores compared to SVG
- **Sparsity**: Tested at 55%, 75%, 80%, 90% sparsity ratios
- **Models**: Evaluated on HunyuanVideo-T2V (768p, 128 frames) and Wan2.1-T2V (512p/768p, 80 frames)

## Technical Details
- Uses average pooling (not max pooling) for better quality
- Block Sparse Attention implementation
- First 25% denoising steps use full attention for quality preservation
- Compatible with existing efficient attention frameworks (FlashAttention, Block Sparse Attention)