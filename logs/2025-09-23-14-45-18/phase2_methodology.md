# Phase 2: Methodology Extraction

## Overview
Compact Attention is a training-free sparse attention framework that integrates offline configuration search with efficient attention computation while preserving video generation fidelity.

## 1. Pattern Analysis Framework

### 1.1 Spatio-Temporal Pattern Discovery
The methodology systematically analyzes attention maps in video diffusion transformers to identify structured patterns:

**Spatial Patterns:**
- **Local Patterns**: Focus on compact neighborhoods around target positions, forming spherical attention fields
- **Cross-Shaped Patterns**: Directional sensitivity creating continuous attention corridors along horizontal/vertical axes
- **Global Patterns**: Full spatial connectivity irrespective of distance, often clustered around salient objects

**Temporal Patterns:**
- **Time-Variant**: Strong correlation with temporal relative distance (progressive decay or specific distance focus)
- **Time-Invariant**: Frame-agnostic distributions maintaining consistent focus across all timesteps

### 1.2 Pattern Stability Analysis
- **Input Invariance**: Patterns remain stable across different text prompts and random seeds (similarity > 0.8)
- **Temporal Robustness**: Configurations stable within denoising step ranges, enabling offline pre-computation

## 2. Tile-Based Deformable Sparse Pattern

### 2.1 Hierarchical Token Organization
Instead of rigid predefined windows, tokens are reorganized into spacetime tiles - clusters adjacent in both spatial and temporal domains.

**Key Components:**
- **Frame-Group-wise Patterns**: Frames partitioned into distance-based groups relative to current frame
- **Dual Attention Windows**: Within each group, spatial masks composed from two complementary window shapes
- **Dynamic Adaptation**: Patterns adapt across spatial and temporal dimensions without explicit classification

### 2.2 Computational Benefits
- **Spatial Adaptability**: Tile combinations emulate diverse attention modes
- **Temporal Awareness**: Distance-stratified configurations
- **Hardware Efficiency**: Preserves computational regularity of tile-based processing

## 3. Optimized Auto-Search Algorithm

### 3.1 Offline Configuration Pipeline
Decouples pattern discovery from runtime execution through offline optimization:

**Process Flow:**
1. Start with full attention coverage
2. Iteratively tighten window boundaries across spatial dimensions
3. Prioritize regions with lower recall contributions
4. Governed by dual thresholds: recall threshold τ and cost threshold λ

### 3.2 Mask Optimization Strategy
- **Boundary Contraction**: Directional shrinkage along hierarchical dimensions
- **Dual Threshold Control**: 
  - τ: Minimum recall threshold preserving critical interactions
  - λ: Maximum cost threshold balancing reduction vs accuracy
- **Termination Conditions**: Recall drops below τ OR cost ratio exceeds λ

### 3.3 Cross-Prompt Merging
- **Conservative Union Operations**: Merge configurations across prompts
- **Guaranteed Coverage**: Retains all potentially relevant attention regions
- **Configuration Caching**: Reuse masks across n consecutive denoising steps

## 4. Implementation Details

### 4.1 Computational Framework
- **ThunderKittens-based**: Efficient GPU kernel implementation
- **Tile-Level Processing**: Blocks of size bq, bk for queries and keys
- **Online Softmax**: Memory-efficient incremental computation

### 4.2 Pattern Classification Criteria
- **Global Pattern**: >85% spatial coverage
- **Local Pattern**: Focus within axes-aligned constraints defined by boundaries ω, η
- **Cross-Shaped**: Complementary axis dominance with (ω1-ω2)(η1-η2) < 0

### 4.3 Parameter Selection
- **Recall Threshold (τ)**: 0.9 for optimal quality
- **Cost Threshold (λ)**: 0.011 for Wan, 0.04 for Hunyuan
- **Step Reuse (n)**: Configurations cached across denoising steps

## 5. Runtime Complexity Analysis

### 5.1 Baseline Full Attention
- **Computation**: [N, N, d] where N = f×h×w tokens
- **Memory**: O(N²) attention matrix storage
- **Time**: O(N²d) for attention computation

### 5.2 Compact Attention
- **Sparsity Rate**: Up to 62.36% (Hunyuan)
- **Effective Computation**: [N, k, d] where k = (1-sparsity)×N
- **Memory**: O(Nk) for sparse attention
- **Time**: O(Nkd) with tile-level optimization

### 5.3 Communication Overhead
- **Offline Cost**: One-time mask generation per model-layer-head combination
- **Runtime Overhead**: Negligible (pre-computed masks loaded from cache)
- **Storage**: O(N) for mask configurations