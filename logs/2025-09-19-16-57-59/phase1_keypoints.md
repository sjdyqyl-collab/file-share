# Phase 1: Key Points Extraction - DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance

## Problem Statement
- **Computational Bottleneck**: Attention mechanism accounts for over 80% of total latency in video diffusion transformers
- **Scale Challenge**: Generating 8 seconds of 720p video takes tens of minutes
- **Quadratic Complexity**: Attention has O(n²) complexity with respect to context length

## Core Innovation - DraftAttention Framework
- **Training-free acceleration** for video diffusion transformers
- **Dynamic sparse attention** guided by low-resolution draft attention maps
- **Hardware-friendly execution** through deterministic reordering

## Key Technical Contributions

### 1. Low-Resolution Draft Attention
- **Downsampling Strategy**: Uses 8×16 average pooling kernel with stride=kernel size
- **Token Reduction**: Reduces tokens by factor of 128 (8×16=128)
- **Efficiency**: Minimal computational overhead for draft attention computation

### 2. Sparse Attention Mechanism
- **Two-stage process**:
  1. Lightweight draft attention on downsampled features
  2. Guided sparse attention on full-resolution features
- **Sparsity Pattern**: Top-r selection from draft attention map
- **Structured Sparsity**: Region-level sparsity aligned with token-level computation

### 3. Theoretical Guarantees
- **Error Bounds**: Formal analysis showing controlled approximation error
  - Draft attention error: ∥S-Sdraft∥F ≤ δn
  - Sparsity mask error: ∥S-S⊙cM∥F ≤ n(δ+t)√(1-r)
- **Approximation Quality**: Low-resolution draft attention closely approximates full attention

### 4. Hardware Optimization
- **Deterministic Reordering**: Ensures contiguous memory access for sparse patterns
- **Block-aligned Processing**: 128 tokens per kernel processed as single unit
- **Compatibility**: Works with FlashAttention and Block Sparse Attention frameworks

## Experimental Results
- **Speedup**: Up to 1.75× end-to-end acceleration on H100 GPU
- **Quality Preservation**: Better than Sparse VideoGen (SVG) under same sparsity
- **Scalability**: Tested on HunyuanVideo-T2V (768p, 128 frames) and Wan2.1-T2V (512p/768p, 80 frames)
- **Sparsity Ratios**: 55%, 60%, 75%, 80%, 90% tested with quality metrics

## Advantages Over Existing Methods
- **Dynamic vs Static**: Unlike static patterns in SVG, provides dynamic sparsity per attention module
- **Training-free**: No additional training required (vs AdaSpa's warm-up requirement)
- **Better Quality**: Maintains higher PSNR, SSIM, and lower LPIPS compared to SVG
- **Hardware Efficiency**: Structured sparsity enables GPU optimization

## Practical Impact
- **Plug-and-play**: Can be integrated into existing video diffusion transformers
- **Real-world Applicability**: Addresses major bottleneck in video generation deployment
- **Scalability**: Enables longer/higher-resolution video generation within practical time constraints