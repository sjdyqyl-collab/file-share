# Phase 1: Key Points Extraction

## Title
DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance

## Abstract (Original)
Diffusion transformer-based video generation models (DiTs) have recently attracted widespread attention for their excellent generation quality. However, their computational cost remains a major bottleneck—attention alone accounts for over 80% of total latency, and generating just 8 seconds of 720p video takes tens of minutes—posing serious challenges to practical application and scalability. To address this, we propose the DraftAttention, a training-free framework for the acceleration of video diffusion transformers with dynamic sparse attention on GPUs. We apply down-sampling to each feature map across frames in the compressed latent space, enabling a higher-level receptive field over the latent composed of hundreds of thousands of tokens. The low-resolution draft attention map, derived from draft query and key, exposes redundancy both spatially within each feature map and temporally across frames. We reorder the query, key, and value based on the draft attention map to guide the sparse attention computation in full resolution, and subsequently restore their original order after the attention computation. This reordering enables structured sparsity that aligns with hardware-optimized execution. Our theoretical analysis demonstrates that the low-resolution draft attention closely approximates the full attention, providing reliable guidance for constructing accurate sparse attention. Experimental results show that our method outperforms existing sparse attention approaches in video generation quality and achieves up to 1.75× end-to-end speedup on GPUs.

## Key Contributions
1. **Vision-centric approach**: Using pooling to extract high-level representations with broader receptive field for identifying spatial and temporal redundancy in video diffusion
2. **DraftAttention framework**: Hardware-friendly approach accelerating video diffusion transformers using guidance from low-resolution draft attention
3. **Theoretical justification**: Formal analysis showing controlled difference between full-resolution and low-resolution draft attention, with bounded error
4. **Experimental validation**: Better video generation quality compared to other sparse attention methods, achieving up to 1.75× end-to-end acceleration on GPUs

## Core Problem
- Video diffusion transformers (DiTs) have quadratic complexity O(n²) in sequence length
- Attention mechanism accounts for >80% of total latency
- Generating 8s 720p video takes tens of minutes
- Existing sparse attention methods have static patterns or significant quality degradation

## Proposed Solution
- Two-stage attention: lightweight draft attention → masked sparse attention
- Low-resolution draft attention via average pooling (8×16 kernel, stride=kernel size)
- Region-level sparsity pattern lifted to token resolution
- Deterministic reordering for hardware-friendly execution
- Training-free, plug-and-play integration

## Technical Innovation
- Dynamic sparse attention at per-module level
- Structured sparsity through region-based masking
- Memory layout optimization via deterministic reordering
- Theoretical bounds on approximation error

## Performance Results
- Up to 1.75× end-to-end speedup on H100 GPUs
- Better generation quality than Sparse VideoGen (SVG) under same sparsity
- Maintains quality at 90% sparsity ratio
- Validated on HunyuanVideo and Wan2.1 models

## Key Parameters
- Pooling kernel: 8×16 with stride=kernel size (128× token reduction)
- Sparsity ratios tested: 55%, 60%, 75%, 80%, 90%
- Resolutions: 512p and 768p
- Models: HunyuanVideo-T2V (128 frames), Wan2.1-T2V (80 frames)