# Phase 1: Key Points Extraction - Compact Attention Paper

## Title
Compact Attention: Exploiting Structured Spatio-Temporal Sparsity for Fast Video Generation

## Original Abstract (Preserved)
The computational demands of self-attention mechanisms pose a critical challenge for transformer-based video generation, particularly in synthesizing ultra-long sequences. Current approaches, such as factorized attention and fixed sparse patterns, fail to fully exploit the inherent spatio-temporal redundancies in video data. Through systematic analysis of video diffusion transformers (DiT), we uncover a key insight: Attention matrices exhibit structured, yet heterogeneous sparsity patterns, where specialized heads dynamically attend to distinct spatiotemporal regions (e.g., local pattern, cross-shaped pattern, or global pattern). Existing sparse attention methods either impose rigid constraints or introduce significant overhead, limiting their effectiveness. To address this, we propose Compact Attention, a hardware-aware acceleration framework featuring three innovations: 1) Adaptive tiling strategies that approximate diverse spatial interaction patterns via dynamic tile grouping, 2) Temporally varying windows that adjust sparsity levels based on frame proximity, and 3) An automated configuration search algorithm that optimizes sparse patterns while preserving critical attention pathways. Our method achieves 1.6∼2.5×acceleration in attention computation on single-GPU setups while maintaining comparable visual quality with full-attention baselines. This work provides a principled approach to unlocking efficient long-form video generation through structured sparsity exploitation.

## Key Findings and Contributions

### 1. Fundamental Discovery
- **Structured Sparsity Patterns**: Video diffusion transformers exhibit heterogeneous sparsity patterns that are stable across inputs
- **Specialized Attention Heads**: Different heads focus on distinct spatiotemporal regions:
  - Local patterns (fine-grained details)
  - Cross-shaped patterns (directional sensitivity)
  - Global patterns (full spatial connectivity)
  - Time-variant patterns (temporal relative distance)
  - Time-invariant patterns (frame-agnostic distributions)

### 2. Key Insights
- **Pattern Stability**: Attention patterns are stable across different prompts, seeds, and within denoising step ranges
- **3D Redundancy**: Video data has inherent 3D spatio-temporal redundancies not exploited by existing methods
- **Tile-Based Efficiency**: Block-wise processing with 3D spatial grouping improves sparsity by 1-3.4% over direct flattening

### 3. Main Contributions
1. **Pattern Discovery**: Revealed structured hierarchical attention patterns in video diffusion transformers
2. **Compact Attention Framework**: Training-free sparse attention with offline configuration search
3. **Hardware-Aware Design**: Tile-based computation optimized for heterogeneous sparsity structures
4. **Performance Results**: 2.5× speedup on Hunyuan model, 1.65× on Wan2.1 model with negligible quality loss

### 4. Technical Innovations
- **Adaptive Tiling**: Dynamic tile grouping for diverse spatial patterns
- **Temporal Windows**: Frame-proximity based sparsity adjustment
- **Auto-Search Algorithm**: Offline optimization preserving critical pathways
- **Dual Attention Windows**: Cross-shaped and local pattern approximation
- **Frame-Group Patterns**: Distance-based temporal grouping

### 5. Performance Metrics
- **Hunyuan (127K tokens)**: 62.36% sparsity, 2.51× speedup, PSNR 30.08
- **Wan2.1 (80K tokens)**: 33.99% sparsity, 1.65× speedup, PSNR 23.73
- **Quality Preservation**: SSIM > 0.77, minimal degradation on VBench metrics

### 6. Limitations Identified
- Auto-search strategy may compromise visual fidelity under aggressive thresholds
- Critical details might be omitted in demanding scenarios
- Need for adaptive thresholding and context-aware strategies