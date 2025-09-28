# Phase 2: Methodology Extraction - Compact Attention Framework

## Method Overview
Compact Attention is a hardware-aware acceleration framework that exploits structured spatio-temporal sparsity in video diffusion transformers through three key innovations.

## 1. Tile-Based Sparsity Foundation
### Block-wise Attention Processing
- Processes data in blocks rather than token-by-token to reduce overhead
- Exploits clustering characteristics of sparsity in 3D space
- Uses spatially adjacent tiles in 3D space (f, h, w) instead of 1D flattening
- Reduces active blocks by 1.1% (Wan2.1) and 3.4% (Hunyuan) while maintaining 95% recall

### Token Reordering Strategy
- Groups tokens based on 3D spatial locality before applying block-wise sparsity
- Maintains acceleration benefits while improving sparsity
- Compatible with FlashAttention-style tile-based computation

## 2. Spatio-Temporal Pattern Analysis
### Spatial Patterns Identified
1. **Local Patterns**: Focus on compact neighborhoods around target positions
   - Defined by: R_local = {(x,y) | max(|x-x_t|/ω, |y-y_t|/η) ≤ 1}
   - Forms spherical attention fields for fine-grained detail synthesis

2. **Cross-Shaped Patterns**: Directional sensitivity along horizontal/vertical axes
   - Defined by: R_cross = ∪_{k=1}^2 (|x-x_t|/ω_k ≤ 1 ∧ |y-y_t|/η_k ≤ 1)
   - Where (ω_1-ω_2)(η_1-η_2) < 0 enforces complementary axis dominance

3. **Global Patterns**: Full spatial connectivity regardless of distance
   - Input-dependent clustering around salient objects

### Temporal Patterns Identified
1. **Time-Variant Patterns**: Strong correlation with temporal relative distance
   - Progressive weight decay across frames
   - Focus on specific frame distances

2. **Time-Invariant Patterns**: Frame-agnostic distributions
   - Consistent focus across all timesteps

## 3. Compact Attention Framework Components

### 3.1 Tile-based Deformable Sparse Pattern
**Frame-Group-wise Patterns**:
- Partition frames into distance-based groups relative to current frame
- Each group has its own sparse configuration
- Captures temporal dynamics through distance-stratified configurations

**Dual Attention Windows**:
- Adaptively compose spatial attention masks from two complementary window shapes
- Approximates observed patterns (cross-shaped, local blocks)
- Eliminates need for explicit pattern classification during inference

### 3.2 Optimized Auto-Search Algorithm
**Offline Configuration Pipeline**:
- Decouples pattern discovery from runtime execution
- Preserves spatiotemporal coherence through offline pre-computation
- Pattern stability enables reliable offline optimization

**Boundary Contraction Process**:
1. Starts with full attention coverage
2. Iteratively tightens window boundaries across spatial dimensions
3. Prioritizes regions with lower recall contributions
4. Operates independently across different frame groups

**Dual Threshold System**:
- **Recall threshold (τ)**: Minimum recall to preserve critical interactions (typically 0.9)
- **Cost threshold (λ)**: Maximum cost to balance computational reduction (e.g., 0.011 for Wan, 0.04 for Hunyuan)

**Configuration Merging**:
- Merges configurations across prompts through union operations
- Conservative strategy retains all potentially relevant attention regions
- Reduces search frequency by n× through configuration caching across denoising steps

## 4. Pattern Stability Properties
### Input Invariance
- Pattern sizes remain stable across text prompts and random seeds
- Average similarity > 0.8 measured by: Sim(M_A, M_B) = ||M_A⊙M_B||_1 / ||M_A+M_B-M_A⊙M_B||_1

### Temporal Robustness
- Attention configurations stable within denoising step ranges
- Enables reliable offline pre-computation per model-layer-head combination

## 5. Implementation Details
### Hardware Awareness
- Based on ThunderKittens framework
- Tile abstraction aligns with GPU memory hierarchy
- Maintains computational regularity for hardware efficiency

### Training-Free Operation
- No additional training required
- Offline mask configuration search
- Runtime application of pre-computed masks

### Quality Preservation Strategy
- Full attention in early denoising steps (first 15 steps)
- Sparse attention in later steps for acceleration
- Maintains structural initialization quality while enabling speedup