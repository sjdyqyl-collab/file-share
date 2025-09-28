# Compact Attention: Methodology

## Overview
Compact Attention is a hardware-aware acceleration framework that exploits structured sparsity in video diffusion transformers through three key innovations: adaptive tiling strategies, temporally varying windows, and automated configuration search.

## Core Methodology

### 1. Pattern Analysis and Discovery

#### 1.1 Spatio-Temporal Pattern Identification
- **3D Analysis**: Examines attention patterns in (f, h, w) space
- **Query-specific Analysis**: Studies per-query-token attention distributions
- **Pattern Types**:
  - **Spatial**: Local (spherical neighborhoods), Cross-shaped (horizontal/vertical corridors), Global (full connectivity)
  - **Temporal**: Time-variant (distance-dependent), Time-invariant (frame-agnostic)

#### 1.2 Pattern Stability Analysis
- **Input Invariance**: Patterns remain stable across different text prompts and random seeds
- **Temporal Robustness**: Stable across denoising steps within certain ranges
- **Layer Consistency**: Consistent patterns across different layers and heads

### 2. Tile-Based Deformable Sparse Pattern

#### 2.1 Tile Abstraction
- **3D Tiles**: Groups tokens into spacetime tiles (adjacent in spatial and temporal domains)
- **Hierarchical Grouping**: Respects video's dual nature (temporal variation + spatial locality)
- **Computational Unit**: Tiles serve as basic building blocks for attention computation

#### 2.2 Frame-Group-wise Patterns
- **Distance-based Grouping**: Partitions frames relative to current processing frame
- **Group-specific Masks**: Each frame group has its own sparse configuration
- **Temporal Adaptation**: Captures temporal dynamics through varying sparsity levels

#### 2.3 Dual Attention Windows
- **Complementary Shapes**: Combines cross-shaped and local block patterns
- **Dynamic Composition**: Adaptively composes spatial attention masks
- **Pattern Approximation**: Eliminates need for explicit pattern classification

### 3. Automated Mask Search Algorithm

#### 3.1 Offline Configuration Pipeline
- **Decoupled Design**: Separates pattern discovery from runtime execution
- **Pre-computation**: Optimizes masks offline to avoid runtime overhead
- **Cross-prompt Merging**: Uses union operations to merge configurations across prompts

#### 3.2 Boundary Contraction Process
- **Iterative Shrinkage**: Starts with full attention coverage
- **Hierarchical Tightening**: Tightens window boundaries across spatial dimensions
- **Cost-Recall Trade-off**: Balances computational reduction against accuracy loss

#### 3.3 Dual Threshold System
- **Recall Threshold (τ)**: Minimum recall to preserve critical interactions
- **Cost Threshold (λ)**: Maximum cost threshold for computational reduction
- **Termination Criteria**: Stops when recall < τ or cost ratio exceeds λ

#### 3.4 Configuration Caching
- **Temporal Reuse**: Reuses masks across n consecutive denoising steps
- **Frequency Reduction**: Reduces search frequency by n×
- **Quality Preservation**: Maintains generation quality through stable patterns

## Technical Implementation

### 1. Computational Flow
1. **Offline Phase**:
   - Analyze attention patterns across model layers/heads
   - Search optimal sparse masks using boundary contraction
   - Cache configurations for runtime use

2. **Runtime Phase**:
   - Load pre-computed masks for current layer/head
   - Apply tile-based sparse attention computation
   - Maintain quality through preserved critical pathways

### 2. Hardware Optimization
- **Tile-based Processing**: Aligns with GPU memory hierarchy
- **Block-wise Computation**: Uses FlashAttention-style tiling
- **Memory Efficiency**: Reduces memory access through sparsity

### 3. Mathematical Formulation

#### 3.1 Local Pattern Definition
```
R_local = {(x,y) | max(|x-x_t|/ω, |y-y_t|/η) ≤ 1}
```

#### 3.2 Cross-shaped Pattern Definition
```
R_cross = ∪_{k=1}^2 {(x,y) | |x-x_t|/ω_k ≤ 1 ∧ |y-y_t|/η_k ≤ 1}
```
with (ω_1-ω_2)(η_1-η_2) < 0 for complementary axis dominance

#### 3.3 Pattern Similarity Metric
```
Sim(M_A, M_B) = ||M_A ⊙ M_B||_1 / ||M_A + M_B - M_A ⊙ M_B||_1
```

## Key Design Decisions

### 1. Tile Size Selection
- **Trade-off**: Larger tiles = better hardware efficiency, smaller tiles = better pattern approximation
- **Empirical**: Chosen based on GPU architecture and model characteristics

### 2. Frame Group Granularity
- **Temporal Distance**: Groups frames based on relative distance to current frame
- **Group Count**: Balances temporal adaptation with computational overhead

### 3. Threshold Selection
- **Recall (τ)**: Typically 0.9 to maintain quality
- **Cost (λ)**: Model-specific, tuned for optimal speed-quality trade-off

## Complexity Analysis

### Time Complexity
- **Full Attention**: O(n²) where n = number of tokens
- **Compact Attention**: O(n² × sparsity_ratio) + O(mask_lookup)
- **Mask Lookup**: O(1) with pre-computed masks

### Space Complexity
- **Mask Storage**: O(L × H × G × T²) where L=layers, H=heads, G=groups, T=tile_size
- **Runtime Memory**: Reduced by sparsity ratio compared to full attention

## Implementation Details
- **Framework**: Built on ThunderKittens with FlashAttention-2
- **GPU Target**: Optimized for H800 GPU
- **Integration**: Compatible with diffusers library
- **Models**: Tested on Wan2.1 (14B) and Hunyuan video diffusion models