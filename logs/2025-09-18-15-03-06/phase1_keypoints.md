# DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance - Key Points

## Original Abstract (Retained)
Diffusion transformer–based video generation models (DiTs) have recently attracted widespread attention for their excellent generation quality. However, their computational cost remains a major bottleneck—attention alone accounts for over 80% of total latency, and generating just 8 seconds of 720p video takes tens of minutes—posing serious challenges to practical application and scalability. To address this, we propose the DraftAttention, a training-free framework for the acceleration of video diffusion transformers with dynamic sparse attention on GPUs. We apply down-sampling to each feature map across frames in the compressed latent space, enabling a higher-level receptive field over the latent composed of hundreds of thousands of tokens. The low-resolution draft attention map, derived from draft query and key, exposes redundancy both spatially within each feature map and temporally across frames. We reorder the query, key, and value based on the draft attention map to guide the sparse attention computation in full resolution, and subsequently restore their original order after the attention computation. This reordering enables structured sparsity that aligns with hardware-optimized execution. Our theoretical analysis demonstrates that the low-resolution draft attention closely approximates the full attention, providing reliable guidance for constructing accurate sparse attention. Experimental results show that our method outperforms existing sparse attention approaches in video generation quality and achieves up to 1.75× end-to-end speedup on GPUs.

## Key Technical Contributions

### 1. DraftAttention Framework
- **Training-free acceleration** of video diffusion transformers
- **Dynamic sparse attention** guided by low-resolution draft attention maps
- **Hardware-friendly implementation** through token reordering
- **Plug-and-play integration** with existing models

### 2. Core Innovation
- **Two-stage attention mechanism**:
  1. Lightweight draft attention on downsampled features
  2. Guided sparse attention on full-resolution features
- **Average pooling-based downsampling** (8×16 kernel with stride=kernel size)
- **128× token reduction** in draft attention computation
- **Deterministic reordering** for contiguous memory access

### 3. Theoretical Guarantees
- **Error bounds** for draft attention approximation
- **Controlled difference** between full and low-resolution attention
- **Bounded sparsity-induced error** with theoretical justification

### 4. Performance Achievements
- **1.75× end-to-end speedup** on H100 GPUs
- **Superior generation quality** compared to Sparse VideoGen
- **90% sparsity** achievable with minimal quality degradation
- **Maintained perceptual quality** across multiple metrics (PSNR, SSIM, LPIPS)

## Critical Observations
- Attention mechanism accounts for >80% of total latency in video DiTs
- Quadratic complexity with sequence length is the primary bottleneck
- Static sparse patterns show significant quality degradation
- Dynamic per-module sparsity provides better quality-efficiency tradeoff
- Average pooling outperforms max pooling for draft attention generation

## Models Evaluated
- HunyuanVideo-T2V (768p, 128 frames)
- Wan2.1-T2V (512p and 768p, 80 frames)

## Evaluation Metrics
- VBench scores (image quality, subject consistency, background consistency, dynamic degree, aesthetic quality)
- PSNR, SSIM, LPIPS for similarity measurement
- PFLOPs for computational cost
- End-to-end latency on H100 GPU