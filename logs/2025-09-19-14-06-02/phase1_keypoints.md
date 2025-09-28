# Phase 1: Key Points Extraction - DraftAttention Paper

## Core Problem
- Video diffusion transformers (DiTs) suffer from extreme computational costs in attention mechanism
- Attention accounts for >80% of total latency in video generation
- 8 seconds of 720p video takes tens of minutes to generate
- Quadratic complexity with respect to context length becomes bottleneck for long sequences

## Proposed Solution: DraftAttention
- **Training-free framework** for accelerating video diffusion transformers
- Uses **low-resolution draft attention guidance** for dynamic sparse attention
- Key innovation: Downsample feature maps via average pooling to create low-resolution attention maps
- Identifies redundancy both spatially (within frames) and temporally (across frames)

## Technical Approach
1. **Two-stage mechanism**:
   - Stage 1: Compute lightweight draft attention on downsampled query/key
   - Stage 2: Apply guided sparse attention on full-resolution representations

2. **Reordering mechanism**: 
   - Reorders query/key/value based on draft attention map
   - Enables structured sparsity aligned with hardware-optimized execution
   - Restores original order after computation

## Key Advantages
- **Efficiency**: Minimal overhead from low-resolution draft computation
- **Effectiveness**: Preserves essential visual patterns while reducing computation
- **Plug-and-Play**: No training required, integrates seamlessly into existing models
- **Hardware-friendly**: Structured sparsity enables efficient GPU execution

## Theoretical Justification
- Provides bounds on approximation error between full and draft attention
- Shows error introduced by sparse pattern remains controlled
- Demonstrates low-resolution draft attention closely approximates full attention

## Experimental Results
- **Speed**: Up to 1.75× end-to-end acceleration on GPUs
- **Quality**: Outperforms existing sparse attention methods (Sparse VideoGen)
- **Sparsity**: Achieves 90% sparsity with minimal quality degradation
- **Models tested**: HunyuanVideo-T2V (768p, 128 frames), Wan2.1-T2V (512p/768p, 80 frames)

## Technical Details
- Uses 8×16 pooling kernel with stride=kernel size (128× token reduction)
- Block Sparse Attention implementation
- Retains full attention for first 25% of denoising steps for quality preservation
- Tested on H100 GPUs with VBench evaluation metrics