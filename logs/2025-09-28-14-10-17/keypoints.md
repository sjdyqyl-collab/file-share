# Compact Attention: Key Points

## Abstract (Original)
The computational demands of self-attention mechanisms pose a critical challenge for transformer-based video generation, particularly in synthesizing ultra-long sequences. Current approaches, such as factorized attention and fixed sparse patterns, fail to fully exploit the inherent spatio-temporal redundancies in video data. Through systematic analysis of video diffusion transformers (DiT), we uncover a key insight: Attention matrices exhibit structured, yet heterogeneous sparsity patterns, where specialized heads dynamically attend to distinct spatiotemporal regions (e.g., local pattern, cross-shaped pattern, or global pattern). Existing sparse attention methods either impose rigid constraints or introduce significant overhead, limiting their effectiveness. To address this, we propose Compact Attention, a hardware-aware acceleration framework featuring three innovations: 1) Adaptive tiling strategies that approximate diverse spatial interaction patterns via dynamic tile grouping, 2) Temporally varying windows that adjust sparsity levels based on frame proximity, and 3) An automated configuration search algorithm that optimizes sparse patterns while preserving critical attention pathways. Our method achieves 1.6∼2.5× acceleration in attention computation on single-GPU setups while maintaining comparable visual quality with full-attention baselines. This work provides a principled approach to unlocking efficient long-form video generation through structured sparsity exploitation.

## Key Contributions

### 1. Structured Sparsity Discovery
- **Insight**: Video diffusion transformers exhibit structured, hierarchical attention patterns
- **Spatial Patterns**: Local, cross-shaped, and global patterns
- **Temporal Patterns**: Time-variant (frame-distance dependent) and time-invariant patterns
- **Stability**: Patterns are stable across different inputs and denoising steps

### 2. Compact Attention Framework
- **Training-free**: No retraining required
- **Hardware-aware**: Optimized for GPU efficiency
- **Three key innovations**:
  1. Tile-based deformable sparse patterns
  2. Frame-group-wise temporal adaptation
  3. Automated offline mask search algorithm

### 3. Performance Achievements
- **Speedup**: 1.6-2.5× acceleration on single GPU
- **Quality**: Maintains comparable visual quality to full attention
- **Models tested**: Wan2.1 (14B) and Hunyuan
- **Sparsity**: Up to 62.36% sparsity with minimal quality loss

## Technical Highlights
- **Tile-based computation**: Groups tokens into 3D spacetime tiles for hardware efficiency
- **Dual attention windows**: Combines cross-shaped and local patterns
- **Offline optimization**: Pre-computes masks to avoid runtime overhead
- **Pattern stability**: Leverages stable attention patterns across inputs and timesteps

## Problem Addressed
- **Challenge**: Quadratic complexity of self-attention in video generation
- **Example**: 128-frame 720p video requires processing 100K+ tokens
- **Current limitations**: Existing sparse methods either too rigid or introduce overhead