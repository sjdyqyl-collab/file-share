# Compact Attention: Exploiting Structured Spatio-Temporal Sparsity for Fast Video Generation

## Abstract
The computational demands of self-attention mechanisms pose a critical challenge for transformer-based video generation, particularly in synthesizing ultra-long sequences. Current approaches, such as factorized attention and fixed sparse patterns, fail to fully exploit the inherent spatio-temporal redundancies in video data. Through systematic analysis of video diffusion transformers (DiT), we uncover a key insight: Attention matrices exhibit structured, yet heterogeneous sparsity patterns, where specialized heads dynamically attend to distinct spatiotemporal regions (e.g., local pattern, cross-shaped pattern, or global pattern). Existing sparse attention methods either impose rigid constraints or introduce significant overhead, limiting their effectiveness. To address this, we propose Compact Attention, a hardware-aware acceleration framework featuring three innovations: 1) Adaptive tiling strategies that approximate diverse spatial interaction patterns via dynamic tile grouping, 2) Temporally varying windows that adjust sparsity levels based on frame proximity, and 3) An automated configuration search algorithm that optimizes sparse patterns while preserving critical attention pathways. Our method achieves 1.6∼2.5×acceleration in attention computation on single-GPU setups while maintaining comparable visual quality with full-attention baselines. This work provides a principled approach to unlocking efficient long-form video generation through structured sparsity exploitation.

## 1. Introduction
Transformer-based video generation faces a fundamental computational bottleneck when processing ultra-long sequences. For instance, generating a 128-frame 720p HD video using the Hunyuan-video architecture requires processing over 100K tokens, with attention computation consuming 68-72% of total generation time. While attention matrices exhibit significant sparsity, existing methods fail to effectively leverage the structured spatio-temporal patterns unique to video data.

Our systematic analysis reveals that attention heads in video diffusion transformers exhibit specialized behaviors: some focus on local spatial regions, others form cross-shaped spatial interactions, and certain heads maintain global connectivity. Additionally, temporal patterns show both time-variant (distance-dependent) and time-invariant behaviors. These structured patterns present opportunities for efficient approximation, but current approaches impose rigid constraints or introduce significant overhead.

## 2. Methodology

### 2.1 Sparsity Pattern Discovery

**Spatial Patterns:**
- **Local Patterns**: Compact spherical neighborhoods around target positions defined as:
  `Rlocal = {(x,y) | max(|x-xt|/ω, |y-yt|/η) ≤ 1}`
- **Cross-Shaped Patterns**: Horizontal and vertical attention corridors:
  `Rcross = ∪_{k=1}^2 (|x-xt|/ωk ≤ 1 ∧ |y-yt|/ηk ≤ 1)`
- **Global Patterns**: Full spatial connectivity regardless of distance

**Temporal Patterns:**
- **Time-Variant**: Strong correlation with temporal relative distance
- **Time-Invariant**: Frame-agnostic distributions maintaining consistent focus

### 2.2 Compact Attention Framework

**Tile-Based Deformable Sparse Pattern:**
- Hierarchical grouping using spacetime tiles (clusters adjacent in spatial and temporal domains)
- Frame-group-wise patterns for temporal dynamics
- Dual attention windows for spatial pattern approximation
- Preserves computational regularity for hardware efficiency

**Automated Configuration Search:**
- Offline pipeline decoupling pattern discovery from runtime execution
- Boundary contraction process with dual thresholds:
  - Recall threshold (τ=0.9): preserves critical interactions
  - Cost threshold (λ=0.011 for Wan, 0.04 for Hunyuan): balances computation vs accuracy
- Union operations across prompts for conservative merging
- Configuration caching across denoising steps

## 3. Experiments

### 3.1 Setup
- **Hardware**: Single H800 GPU
- **Models**: Wan2.1 (14B) and Hunyuan
- **Resolution**: 768×1280
- **Frames**: 81 (Wan2.1), 129 (Hunyuan)

### 3.2 Results

**Performance Comparison:**

| Model | Method | Sparsity | PSNR | Latency(s) | Speedup |
|-------|--------|----------|------|------------|---------|
| Wan2.1 | Full Attention | 0% | - | 1092.168 | 1.00x |
| Wan2.1 | Compact Attention | 33.99% | 23.730 | 663.824 | 1.65x |
| Hunyuan | Full Attention | 0% | - | 1370.658 | 1.00x |
| Hunyuan | Compact Attention | 62.36% | 30.082 | 546.504 | 2.51x |

**Quality Metrics (Hunyuan):**
- Subject Consistency: 0.9716 (vs 0.9736 full attention)
- Background Consistency: 0.9693 (vs 0.9735 full attention)
- Aesthetic Quality: 0.6531 (vs 0.6542 full attention)
- CLIPSIM: 0.2184 (vs 0.2181 full attention)

### 3.3 Ablation Studies

**Pattern Effectiveness:**
- Dual Attention Windows improve cross pattern sparsity by 10%
- Frame-group-wise patterns add 3% improvement
- Overall sparsity improvement: 27.1% with optimized parameters

**Sensitivity Analysis:**
- Early denoising steps critical: 1.02dB PSNR drop if sparse attention applied too early
- Optimal: Full attention for first 15 denoising steps, then sparse attention
- Stable performance across different inputs (94.4% pattern similarity)

## 4. Limitations and Future Work

**Current Limitations:**
- Auto-search with low thresholds may compromise visual fidelity
- Critical visual details might be omitted in demanding scenarios
- Fixed thresholds may not adapt to varying content complexity

**Future Directions:**
- Adaptive thresholding based on content complexity
- Context-aware search strategies
- Extension to multimodal generation systems
- Real-time streaming applications

## 5. Conclusion

Compact Attention provides a principled framework for exploiting structured spatio-temporal sparsity in video diffusion transformers. By identifying and leveraging heterogeneous attention patterns through tile-based computation and automated configuration search, we achieve 1.6-2.5× acceleration while maintaining visual quality. This work establishes a foundation for efficient long-form video generation and offers insights into the specialized roles of attention heads in video transformers.

## Runtime Analysis

**Baseline Method (Full Attention):**
- Computation time: [100K, 100K, 1] matrix multiplication for attention
- Memory complexity: O(n²) where n = sequence length
- Hunyuan 127K tokens: ~1370.658s latency

**Proposed Method (Compact Attention):**
- Computation time: [100K, 62.36K, 1] matrix multiplication (62.36% sparsity)
- Memory complexity: O(n² × sparsity_rate)
- Hunyuan 127K tokens: ~546.504s latency (2.51× speedup)

**Improved Method (Future Enhancement):**
- Computation time: [100K, 70K, 1] matrix multiplication (70% sparsity with adaptive thresholds)
- Additional optimizations: Early-step full attention + adaptive pattern refinement
- Projected runtime: ~400-450s for Hunyuan (3.0-3.4× speedup)