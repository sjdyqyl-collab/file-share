# Phase 1: Key Points Extraction - DraftAttention Paper

## Core Problem
- Video diffusion transformers (DiTs) have excellent quality but high computational cost
- Attention mechanism accounts for 80%+ of total latency in video generation
- Generating 8 seconds of 720p video takes tens of minutes

## Key Innovation - DraftAttention
- **Training-free framework** for accelerating video diffusion transformers
- Uses **low-resolution attention guidance** for dynamic sparse attention on GPUs
- **Two-stage approach**:
  1. Compute low-resolution draft attention map via downsampling
  2. Use draft map to guide sparse attention in full resolution

## Technical Approach
- **Downsampling**: Apply average pooling to feature maps across frames
- **Reordering**: Reorder query/key/value based on draft attention map
- **Hardware optimization**: Structured sparsity aligned with GPU execution
- **Theoretical guarantees**: Bounded error between full and draft attention

## Key Results
- **Speed**: Up to 1.75× end-to-end speedup on GPUs
- **Quality**: Outperforms existing sparse attention methods
- **Sparsity**: Supports 90% sparsity ratio with minimal quality loss
- **Models tested**: HunyuanVideo-T2V (768p, 128 frames), Wan2.1-T2V (512p/768p, 80 frames)

## Theoretical Contributions
- **Error bounds**: Formal analysis showing controlled difference between full and draft attention
- **Frobenius norm bounds**: Quantified approximation error from pooling and sparsification
- **Practical justification**: Error remains bounded under local smoothness assumptions

## Method Advantages
1. **Efficiency**: Lightweight computation on reduced tokens
2. **Effectiveness**: Preserves essential visual patterns
3. **Plug-and-play**: No training required, integrates directly
4. **Hardware-friendly**: Structured memory access patterns

## Comparison Baselines
- **Sparse VideoGen (SVG)**: Static sparse patterns
- **AdaSpa**: Dynamic but prompt-level sparsity
- **Full attention**: Dense baseline for quality comparison

## Technical Specifications
- **Pooling kernel**: 8×16 with stride=kernel size (128× token reduction)
- **Sparsity ratios**: 55%, 60%, 75%, 80%, 90%
- **Metrics**: VBench, PSNR, SSIM, LPIPS, PFLOPs
- **Hardware**: H100 GPU testing