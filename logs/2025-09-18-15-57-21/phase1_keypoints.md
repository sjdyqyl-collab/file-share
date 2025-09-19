# Phase 1: Key Points Extraction

## Paper Title
DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance

## Main Problem
- Video diffusion transformers (DiTs) have excellent generation quality but suffer from high computational cost
- Attention mechanism accounts for over 80% of total latency
- Generating 8 seconds of 720p video takes tens of minutes
- Quadratic complexity of attention becomes bottleneck for long sequences

## Key Innovation
- **DraftAttention**: Training-free framework for accelerating video diffusion transformers
- Uses low-resolution draft attention to guide sparse attention computation
- Two-stage approach: lightweight draft attention + masked sparse attention

## Core Technical Contributions

### 1. Draft Attention Mechanism
- Downsamples query and key using average pooling (8×16 kernel, stride=kernel size)
- Reduces tokens by factor of 128
- Computes low-resolution attention map to identify important regions
- Minimal computational overhead

### 2. Structured Sparsity
- Retains only fraction r of most salient region-to-region interactions
- Creates binary mask M for region-level sparsity
- Lifts region-level mask to token resolution
- Enables hardware-friendly execution

### 3. Theoretical Analysis
- **Theorem 3.3**: Bounds error from average pooling approximation
- **Theorem 3.5**: Bounds error from sparsity masking
- Shows draft attention closely approximates full attention
- Error remains controlled and bounded

### 4. Reordering Algorithm
- Deterministic token reordering for hardware efficiency
- Groups spatial patches contiguously in memory
- Aligns region-level sparsity with token-level computation
- Enables efficient block-wise indexing and masking

## Experimental Results
- Outperforms existing sparse attention methods (Sparse VideoGen)
- Achieves up to 1.75× end-to-end speedup on GPUs
- Maintains generation quality under high sparsity (90%)
- Tested on HunyuanVideo and Wan2.1 models
- Better PSNR, SSIM, LPIPS metrics compared to baselines

## Key Advantages
1. **Efficiency**: Lightweight computation on reduced tokens
2. **Effectiveness**: Preserves essential visual patterns
3. **Plug-and-Play**: No training required, seamless integration
4. **Hardware-Friendly**: Structured sparsity for optimized execution