# Phase 1: Key Points Extraction - Compact Attention Paper

## Title
Compact Attention: Exploiting Structured Spatio-Temporal Sparsity for Fast Video Generation

## Core Problem
- Transformer-based video generation suffers from quadratic complexity of self-attention
- Generating 128-frame 720p HD video requires processing 100K+ tokens
- Attention computation consumes 68-72% of total generation time
- Current sparse attention methods fail to exploit 3D spatio-temporal redundancies

## Key Insights
- Attention matrices exhibit structured yet heterogeneous sparsity patterns
- Specialized heads dynamically attend to distinct spatiotemporal regions:
  - Local pattern (compact neighborhoods)
  - Cross-shaped pattern (horizontal/vertical corridors)
  - Global pattern (full spatial connectivity)
  - Time-variant pattern (temporal distance correlation)
  - Time-invariant pattern (frame-agnostic distributions)

## Main Contributions
1. **Pattern Discovery**: Revealed structured hierarchical attention patterns in video diffusion transformers
2. **Compact Attention Framework**: Training-free sparse attention with offline configuration search
3. **Hardware-aware Design**: Three innovations:
   - Adaptive tiling strategies via dynamic tile grouping
   - Temporally varying windows based on frame proximity
   - Automated configuration search preserving critical attention pathways

## Performance Results
- Achieves 1.6-2.5× acceleration in attention computation on single-GPU
- Validated on Wan2.1 (14B) and Hunyuan models
- Maintains comparable visual quality with full-attention baselines
- Up to 62.36% sparsity with minimal quality degradation

## Technical Approach
- Tile-based deformable sparse patterns
- Frame-group-wise patterns for temporal dynamics
- Dual attention windows for spatial patterns
- Offline auto-search with recall threshold τ and cost threshold λ
- Pattern stability enables offline pre-computation