# Phase 2: Detailed Methodology

## Problem Formulation
Given a model with n layers L = {l₁, l₂, ..., lₙ}, partition into k disjoint groups P = {P₁, P₂, ..., Pₖ} such that:
- Each Pᵢ is assigned to a separate accelerator card
- Memory footprint S(Pᵢ) ≤ C (cache capacity)
- Layers assigned contiguously in original order
- Minimize k (number of partitions)

## Memory Footprint Calculation
For each layer lⱼ:
```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
```

### Component Details:
1. **Weight Size**: parameters × datatype_size
   - FP16: 2 bytes per parameter
   - Calculated as: num_params × 2

2. **Activation Size**: output_feature_map × batch_size × datatype_size
   - Depends on layer output dimensions and batch size (1024)

3. **Buffer Size**: workspace memory for operators
   - Derived from profiling or analytical models

## Partitioning Algorithms

### 3.1 Greedy Layer Aggregation (Primary Method)
```
Algorithm:
1. Initialize empty partition Pᵢ
2. For each layer lⱼ in order:
   - If S(Pᵢ) + size(lⱼ) ≤ C:
     * Add lⱼ to Pᵢ
   - Else:
     * Finalize Pᵢ with current layers
     * Start new partition Pᵢ₊₁ with lⱼ
3. Continue until all layers assigned
```

### 3.2 Dynamic Programming (Optional Enhancement)
- Optimizes partition boundaries to minimize maximum partition size
- More balanced load distribution
- Higher computational complexity but better utilization

## Deployment Strategy
1. **Pre-deployment**: Calculate size(lⱼ) for all layers
2. **Partitioning**: Apply greedy or DP algorithm to create P₁...Pₖ
3. **Allocation**: Assign each Pᵢ to accelerator card i
4. **Loading**: Load weights and pre-allocate activations/buffers in SRAM/L2
5. **Execution**: Sequential layer execution within each partition
6. **Communication**: Transfer intermediate outputs between partitions only when crossing device boundaries

## Edge Case Handling
- Single layer exceeding cache capacity: apply intra-layer partitioning or compression
- Variable layer sizes: adjust heuristics to prevent under-utilization
- Batch size optimization: reduce activations to fit constraints

## Memory Hierarchy Optimization
- Target: SRAM/L2 cache access (fastest)
- Avoid: DRAM access (slowest)
- Strategy: Fit entire partition in fast memory to eliminate off-chip accesses during layer execution