# Phase 2: Methodology Extraction - Compact Attention

## 1. Tile-Based Sparsity Framework

### 1.1 Core Concept
- **Tile-based computation**: Process tokens in blocks rather than individual tokens
- **3D spatial locality**: Reorganize tokens based on spatial adjacency in (f, h, w) dimensions
- **Block-wise attention**: Use tiles as computational units to reduce overhead

### 1.2 Tile Formation Strategy
```
Input: 3D feature map (f, h, w)
Process: 
1. Group spatially adjacent tokens into tiles
2. Maintain temporal relationships within tiles
3. Apply block-wise sparsity on reorganized sequence
```

### 1.3 Benefits
- **Memory efficiency**: Reduces memory consumption through block processing
- **Computational regularity**: Maintains hardware-friendly computation patterns
- **Sparsity improvement**: 1-3.4% reduction in active blocks compared to direct flattening

## 2. Structured Spatiotemporal Patterns

### 2.1 Spatial Patterns
1. **Local Patterns**: 
   - Focus on compact neighborhoods around target positions
   - Spherical attention fields for fine-grained details
   - Mathematical definition: Rlocal = {(x,y) | max(|x-xt|/ω, |y-yt|/η) ≤ 1}

2. **Cross-Shaped Patterns**:
   - Directional sensitivity along horizontal and vertical axes
   - Continuous attention corridors
   - Mathematical definition: Rcross = ∪_{k=1}^2 (|x-xt|/ωk ≤ 1 ∧ |y-yt|/ηk ≤ 1)

3. **Global Patterns**:
   - Full spatial connectivity irrespective of distance
   - Input-dependent clustering around salient objects

### 2.2 Temporal Patterns
1. **Time-Variant Patterns**:
   - Strong correlation with temporal relative distance
   - Progressive weight decay across frames
   - Focus on specific frame distances

2. **Time-Invariant Patterns**:
   - Frame-agnostic distributions
   - Consistent focus across all timesteps

## 3. Compact Attention Framework

### 3.1 Architecture Overview
```
Input: Video tokens (Q, K, V)
↓
Tile Formation & Grouping
↓
Frame-Group-wise Pattern Selection
↓
Dual Attention Window Application
↓
Sparse Attention Computation
↓
Output: Attention results
```

### 3.2 Tile-based Deformable Sparse Pattern

#### 3.2.1 Frame-Group-wise Patterns
- **Partition strategy**: Divide frames into distance-based groups relative to current frame
- **Group-specific masks**: Each group has its own sparse configuration
- **Temporal awareness**: Different sparsity levels based on frame proximity

#### 3.2.2 Dual Attention Windows
- **Complementary windows**: Combine two window shapes to approximate complex patterns
- **Pattern approximation**: Cross-shaped and local patterns without explicit classification
- **Adaptive composition**: Dynamic combination based on observed patterns

### 3.3 Optimized Auto-Search Algorithm

#### 3.3.1 Offline Configuration Pipeline
```
Algorithm: Mask Auto-Search
Input: Model, Layer, Head, Training prompts
Output: Optimized sparse mask configuration

1. Initialize with full attention coverage
2. Iteratively shrink window boundaries:
   - Prioritize regions with lower recall contributions
   - Directional shrinkage across spatial dimensions
   - Independent operation per frame group
3. Dual threshold control:
   - Recall threshold τ (preserve critical interactions)
   - Cost threshold λ (balance efficiency vs quality)
4. Termination conditions:
   - Recall < τ (quality preservation)
   - ΔRecall/ΔCost > λ (diminishing returns)
5. Cross-prompt merging:
   - Union operation across all prompts
   - Conservative strategy to retain all relevant regions
```

#### 3.3.2 Temporal Stability Exploitation
- **Configuration caching**: Reuse masks across n consecutive denoising steps
- **Search frequency reduction**: n× reduction in search overhead
- **Trajectory stability**: Leverage stable diffusion patterns

## 4. Pattern Stability Analysis

### 4.1 Input Invariance
- **Metric**: Similarity = ||MA⊙MB||₁ / ||MA+MB-MA⊙MB||₁
- **Result**: >0.8 average similarity across prompts and seeds
- **Implication**: Enables offline precomputation

### 4.2 Temporal Robustness
- **Observation**: Patterns stable within denoising step ranges
- **Benefit**: Reliable offline mask optimization per model-layer-head combination
- **Implementation**: Fixed masks for n consecutive steps

## 5. Implementation Details

### 5.1 Hardware Optimization
- **Framework**: ThunderKittens implementation
- **Architecture**: Compatible with FlashAttention-2
- **Memory**: Tile-based processing reduces memory footprint

### 5.2 Parameter Selection
- **Recall threshold (τ)**: 0.9 (quality preservation)
- **Cost threshold (λ)**: 0.011 (Wan2.1), 0.04 (Hunyuan)
- **Step reuse (n)**: Determined empirically based on stability analysis