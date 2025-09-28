# DraftAttention: Key Points Extraction

## Abstract
- **Problem**: Diffusion transformer-based video generation models (DiTs) have excellent quality but high computational cost - attention accounts for 80%+ of total latency
- **Solution**: DraftAttention - a training-free framework for accelerating video diffusion transformers with dynamic sparse attention on GPUs
- **Key Innovation**: Uses down-sampling to create low-resolution draft attention maps that guide sparse attention computation in full resolution
- **Results**: Outperforms existing sparse attention methods, achieves up to 1.75× end-to-end speedup on GPUs

## Core Problem
1. Video generation with DiTs is computationally expensive due to quadratic attention complexity
2. Generating 8 seconds of 720p video takes tens of minutes
3. Attention mechanism alone accounts for >80% of total computation in models like Hunyuan Video

## Key Innovations

### 1. Low-Resolution Draft Attention
- Applies down-sampling via average pooling to feature maps across frames
- Creates higher-level receptive field over latent space with hundreds of thousands of tokens
- Exposes redundancy both spatially within feature maps and temporally across frames
- Computationally lightweight due to reduced token count

### 2. Dynamic Sparse Attention Pattern
- Reorders query, key, and value based on draft attention map
- Guides sparse attention computation in full resolution
- Restores original order after computation
- Enables structured sparsity aligned with hardware-optimized execution

### 3. Theoretical Foundation
- Provides bounds on approximation error between full and draft attention
- Shows low-resolution draft attention closely approximates full attention
- Demonstrates bounded error from sparse pattern derived from draft attention

### 4. Hardware-Friendly Design
- Deterministic reordering aligns region-level sparsity with token-level computation
- Groups scattered sparse patterns into contiguous format
- Enables efficient block-wise processing (128 tokens per kernel)
- Compatible with frameworks like FlashAttention and Block Sparse Attention

## Technical Details

### Methodology
- **Two-stage mechanism**: Lightweight draft attention → masked sparse attention
- **Pooling**: 8×16 kernel with stride equal to kernel size, reducing tokens by factor of 128
- **Sparsity pattern**: Retains fraction r of most salient region-to-region interactions
- **Reordering**: Ensures spatial patches are contiguous in memory

### Theoretical Analysis
- **Draft attention error bound**: ∥S−Sdraft∥F ≤ δn
- **Sparsity mask error bound**: ∥S−S⊙cM∥F ≤ n(δ+t)√1−r
- Where δ is worst-case deviation, t is threshold value, r is sparsity ratio

## Experimental Results
- **Models tested**: HunyuanVideo-T2V (768p, 128 frames), Wan2.1-T2V (512p/768p, 80 frames)
- **Metrics**: VBench, PSNR, SSIM, LPIPS, PFLOPs, latency
- **Performance**: Better quality than Sparse VideoGen (SVG) under same computational budget
- **Speedup**: Up to 1.75× on H100 GPU with 90% sparsity

## Advantages
1. **Efficiency**: Lightweight computation on reduced tokens
2. **Effectiveness**: Captures high-level representations and essential visual patterns
3. **Plug-and-Play**: No training required, integrates seamlessly
4. **Hardware-Friendly**: Structured sparsity for optimized execution

## Limitations Identified
1. Fixed pooling kernel size (8×16) may not be optimal for all scenarios
2. First 25% of denoising steps still use full attention
3. Average pooling may lose fine-grained details
4. No adaptive mechanism for different video content types
5. Limited to specific resolution requirements for optimal performance