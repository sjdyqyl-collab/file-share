# Phase 1: Key Points Extraction

## Paper Title
DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance

## Main Problem
- Diffusion transformer-based video generation models (DiTs) have excellent quality but suffer from high computational cost
- Attention mechanism accounts for over 80% of total latency
- Generating 8 seconds of 720p video takes tens of minutes
- Quadratic complexity with respect to context length becomes a bottleneck for long sequences

## Proposed Solution
- DraftAttention: A training-free framework for accelerating video diffusion transformers
- Uses dynamic sparse attention on GPUs
- Two-stage approach:
  1. Compute low-resolution draft attention map using downsampled query and key
  2. Use draft map to guide sparse attention computation in full resolution

## Key Innovations
1. **Low-resolution draft attention**: Uses average pooling to create downsampled query/key for efficient attention computation
2. **Dynamic sparsity**: Adapts sparse patterns dynamically for each attention module
3. **Hardware-friendly reordering**: Reorders tokens to ensure contiguous memory access for efficient sparse computation
4. **Training-free**: No additional training required, plug-and-play integration

## Technical Details
- Uses 8×16 pooling kernel with stride=kernel size (reduces tokens by 128×)
- Theoretical analysis showing bounded error between full and draft attention
- Deterministic reordering algorithm to align region-level sparsity with token-level computation

## Experimental Results
- Tested on HunyuanVideo-T2V (768p, 128 frames) and Wan2.1-T2V (512p/768p, 80 frames)
- Achieves up to 1.75× end-to-end speedup on H100 GPUs
- Outperforms Sparse VideoGen (SVG) in generation quality under same sparsity ratios
- Maintains better PSNR, SSIM, and LPIPS scores compared to SVG

## Limitations Identified
1. Fixed pooling kernel size (8×16) may not be optimal for all resolutions
2. Static sparsity ratio during inference
3. No consideration of temporal attention patterns beyond spatial pooling
4. Limited to pre-trained models without fine-tuning capabilities
5. Single GPU implementation, no distributed computing support

## Runtime Analysis
- Baseline (full attention): [n, n, d] where n = sequence length, d = hidden dimension
- Proposed method: [g, g, d] + [n, n, d] with sparsity mask where g = n/128 (after 128× reduction)
- Communication time: Not applicable (single GPU)