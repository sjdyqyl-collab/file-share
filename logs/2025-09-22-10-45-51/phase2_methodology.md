# Phase 2: Methodology Extraction - Compact Attention

## Overview
Compact Attention is a hardware-aware acceleration framework that exploits structured spatio-temporal sparsity in video diffusion transformers through three core innovations: adaptive tiling, temporally varying windows, and automated configuration search.

## Core Methodology Components

### 1. Pattern Analysis Foundation
**Structured Sparsity Discovery**
- Analyzed attention maps in video DiT models (Wan2.1, Hunyuan)
- Identified 5 dominant patterns across spatial and temporal dimensions
- Pattern stability validated across prompts, seeds, and denoising steps
- Pattern classification threshold: 85% spatial coverage = global pattern

**Tile-Based Sparsity**
- 3D spatial grouping reduces active blocks by 1-3.4% vs direct flattening
- Block-wise processing for computational efficiency
- Token reorganization based on spatial adjacency in 3D space (f, h, w)

### 2. Compact Attention Framework Architecture

#### 2.1 Tile-Based Deformable Sparse Pattern
**Core Innovation**: Hierarchical grouping respecting video's dual nature

**Frame-Group-wise Patterns**
- Partition frames into distance-based groups relative to current frame
- Each group has independent sparse configuration
- Captures temporal dynamics through stratified configurations

**Dual Attention Windows**
- Complementary window shapes within each frame group
- Approximates observed patterns (cross-shaped, local blocks)
- Eliminates need for explicit pattern classification during inference

**Spatial Adaptability**
- Tile combinations emulate diverse attention modes
- Preserves computational regularity of tile-based processing
- Three-fold synergy: spatial adaptability, temporal awareness, hardware efficiency

#### 2.2 Optimized Auto-Search of Sparse Masks
**Offline Configuration Pipeline**
- Decouples pattern discovery from runtime execution
- Preserves spatiotemporal coherence through conservative merging
- Capitalizes on temporal stability of diffusion trajectories

**Boundary Contraction Process**
- Starts with full attention coverage
- Iteratively tightens window boundaries across spatial dimensions
- Prioritizes regions with lower recall contributions per computational cost
- Independent contraction across different frame groups

**Dual Threshold Control**
- Minimum recall threshold (τ): preserves critical interactions
- Maximum cost threshold (λ): balances computational reduction vs accuracy
- Termination condition: recall < τ OR cost ratio exceeds λ

**Configuration Merging Strategy**
- Union operations across prompts for final configuration
- Conservative merging retains all potentially relevant attention regions
- Mask reuse across n consecutive denoising steps (configuration caching)

### 3. Implementation Details

#### 3.1 Pattern Formulations
**Local Pattern**: Focus around query positions (xt, yt) with axes-aligned constraints
```
Rlocal = {(x, y) | max(|x-xt|/ω, |y-yt|/η) ≤ 1}
```

**Cross-Shaped Pattern**: Complementary spatial constraints
```
Rcross = ∪_{k=1}^2 (|x-xt|/ωk ≤ 1 ∧ |y-yt|/ηk ≤ 1)
```
where (ω1-ω2)(η1-η2) < 0 enforces complementary axis dominance

#### 3.2 Similarity Metrics
**Pattern Similarity**: Sim(MA, MB) = ||MA⊙MB||₁ / ||MA+MB-MA⊙MB||₁
- Average similarity > 0.8 across varying prompts and seeds
- Enables reliable offline pre-computation

#### 3.3 Computational Optimization
- Implemented based on ThunderKittens framework
- Reference to STA framework for compatibility
- FlashAttention-2 integration for memory efficiency
- Tile-level sparsity avoids token-level overhead

### 4. Algorithm Pipeline

#### 4.1 Offline Auto-Search Phase
1. Initialize with full attention coverage
2. Iterative boundary contraction:
   - Calculate recall loss per computational unit
   - Shrink regions with lowest cost-benefit ratio
   - Check dual threshold constraints
3. Cross-prompt configuration merging
4. Cache optimized masks for reuse

#### 4.2 Online Inference Phase
1. Load pre-computed mask configurations
2. Apply frame-group-wise patterns
3. Execute dual attention windows
4. Maintain quality through threshold-controlled sparsity

### 5. Parameter Configuration
**Hunyuan Model**: τ = 0.9, λ = 0.04
**Wan2.1 Model**: τ = 0.9, λ = 0.011
**Temporal Reuse**: n consecutive denoising steps (empirically determined)

### 6. Complexity Analysis
**Time Complexity**: O(n²) → O(n² × sparsity_ratio)
**Space Complexity**: Reduced through tile-based processing
**Memory Efficiency**: Block-wise computation reduces memory footprint

### 7. Hardware Awareness
- Optimized for single-GPU setups
- Compatible with Hopper architecture
- ThunderKittens framework integration
- FlashAttention-2 memory efficiency

## Key Methodological Advantages
1. **Training-Free**: No additional training required
2. **Hardware-Efficient**: Tile-based processing maintains computational regularity
3. **Pattern-Adaptive**: Captures heterogeneous sparsity without rigid constraints
4. **Quality-Preserving**: Dual thresholds balance efficiency and fidelity
5. **Generalizable**: Stable patterns enable offline optimization