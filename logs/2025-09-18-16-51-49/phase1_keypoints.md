# Phase 1: Keypoints Extraction - DraftAttention Paper

## Core Problem
- Video diffusion transformers suffer from massive computational overhead due to attention mechanism
- Attention accounts for >80% of total latency in video generation
- Generating 8 seconds of 720p video takes tens of minutes
- Quadratic complexity of attention with respect to context length becomes bottleneck

## Proposed Solution: DraftAttention
- **Training-free framework** for accelerating video diffusion transformers
- **Dynamic sparse attention** guided by low-resolution draft attention maps
- **Two-stage approach**:
  1. Compute low-resolution draft attention via average pooling
  2. Use draft map to guide sparse attention computation at full resolution

## Key Technical Components
1. **Low-resolution draft attention**: Uses 8×16 pooling kernel with stride=kernel size
2. **Token reordering**: Ensures hardware-friendly execution by grouping sparse patterns
3. **Theoretical guarantees**: Bounded error between full and draft attention
4. **Plug-and-play**: No training required, integrates with existing models

## Performance Results
- **Speedup**: Up to 1.75× end-to-end acceleration on GPUs
- **Quality**: Outperforms existing sparse attention methods (SVG, AdaSpa)
- **Sparsity**: Achieves 90% sparsity with minimal quality degradation
- **Models tested**: HunyuanVideo-T2V (768p, 128 frames), Wan2.1-T2V (512p/768p, 80 frames)

## Theoretical Contributions
- Error bounds for draft attention approximation: ∥S−Sdraft∥F ≤ δn
- Error bounds for sparsity mask: ∥S−S⊙cM∥F ≤ n(δ+t)√(1-r)
- Justification for using low-resolution guidance in attention computation

## Hardware Optimization
- Deterministic reordering algorithm for contiguous memory access
- Compatible with FlashAttention and Block Sparse Attention
- Block-level masking for efficient GPU execution

## Limitations Identified
1. Fixed pooling kernel size (8×16) may not be optimal for all resolutions
2. Static sparsity ratio during inference
3. Limited to spatial pooling, no temporal pooling consideration
4. No adaptive mechanism for different video content types
5. Requires manual tuning of sparsity ratio for different use cases