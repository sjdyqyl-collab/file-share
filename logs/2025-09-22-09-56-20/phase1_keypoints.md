# Phase 1: Key Points Extraction - DraftAttention Paper

## Original Abstract (Preserved)
Diffusion transformer–based video generation models (DiTs) have recently attracted widespread attention for their excellent generation quality. However, their computational cost remains a major bottleneck—attention alone accounts for over 80% of total latency, and generating just 8 seconds of 720p video takes tens of minutes—posing serious challenges to practical application and scalability. To address this, we propose the DraftAttention, a training-free framework for the acceleration of video diffusion transformers with dynamic sparse attention on GPUs. We apply down-sampling to each feature map across frames in the compressed latent space, enabling a higher-level receptive field over the latent composed of hundreds of thousands of tokens. The low-resolution draft attention map, derived from draft query and key, exposes redundancy both spatially within each feature map and temporally across frames. We reorder the query, key, and value based on the draft attention map to guide the sparse attention computation in full resolution, and subsequently restore their original order after the attention computation. This reordering enables structured sparsity that aligns with hardware-optimized execution. Our theoretical analysis demonstrates that the low-resolution draft attention closely approximates the full attention, providing reliable guidance for constructing accurate sparse attention. Experimental results show that our method outperforms existing sparse attention approaches in video generation quality and achieves up to 1.75× end-to-end speedup on GPUs. Code: https://github.com/shawnricecake/draft-attention

## Key Points Summary

### Problem Statement
1. **Computational Bottleneck**: Video diffusion transformers (DiTs) suffer from excessive computational costs, with attention mechanisms consuming over 80% of total latency
2. **Practical Limitations**: Generating 8 seconds of 720p video takes tens of minutes, hindering real-world applications
3. **Quadratic Complexity**: Attention mechanism scales quadratically with sequence length, making long video generation computationally prohibitive

### Proposed Solution: DraftAttention
1. **Training-Free Framework**: No additional training required, works as plug-and-play module
2. **Two-Stage Approach**:
   - Stage 1: Compute low-resolution draft attention map using downsampled queries and keys
   - Stage 2: Use draft map to guide sparse attention computation at full resolution
3. **Dynamic Sparsity**: Adapts sparse patterns dynamically for each attention module
4. **Hardware Optimization**: Uses deterministic reordering for efficient GPU execution

### Technical Innovation
1. **Low-Resolution Guidance**: Uses 8×16 average pooling (128× token reduction) to create draft attention maps
2. **Structured Sparsity**: Groups tokens into contiguous blocks for hardware-friendly computation
3. **Theoretical Bounds**: Provides formal error bounds for both draft attention approximation and sparsity masking
4. **Reordering Mechanism**: Ensures spatial locality and coalesced memory access patterns

### Key Results
1. **Quality Preservation**: Outperforms existing sparse attention methods (SVG) across all metrics
2. **Speed Improvement**: Achieves up to 1.75× end-to-end acceleration on H100 GPUs
3. **Scalability**: Maintains quality across different resolutions (512p, 768p) and models (Wan2.1, HunyuanVideo)
4. **Sparsity Tolerance**: Performs well even at 90% sparsity ratios

### Theoretical Contributions
1. **Error Analysis**: Bounded Frobenius-norm error for draft attention approximation
2. **Sparsity Error**: Controlled error from top-r sparsity masking
3. **Approximation Guarantee**: Formal proof that low-resolution draft attention closely approximates full attention