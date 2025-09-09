# Phase 2: Methodology Extraction

## Problem Formulation
- Input: Model with n layers L = {l₁, l₂, ..., lₙ}
- Output: k disjoint partitions P = {P₁, P₂, ..., Pₖ}
- Constraints: 
  - S(Pᵢ) = Σ size(lⱼ) ≤ C for all partitions
  - Layers assigned contiguously preserving execution order
  - Minimize k for balanced hardware utilization

## Memory Footprint Calculation
For each layer lⱼ:
- size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
- weight_size: parameters × datatype (FP16 = 2 bytes)
- activation_size: output_feature_map × batch_size
- buffer_size: operator workspace from profiling

## Partitioning Algorithms

### 3.1 Greedy Layer Aggregation
```
Input: layers L[1..n], cache capacity C
Output: partitions P[1..k]

Initialize:
  P = []
  current_partition = []
  current_size = 0
  start_layer = 1

For j from 1 to n:
  If current_size + size(L[j]) ≤ C:
    Add L[j] to current_partition
    current_size += size(L[j])
  Else:
    Finalize P_i = {L[start_layer..j-1]}
    Add P_i to P
    Start new partition from L[j]
    current_size = size(L[j])
    start_layer = j

Add final partition to P
Return P
```

### 3.2 Dynamic Programming (Optional)
- Minimize maximum partition size while respecting cache constraints
- More balanced load distribution
- Higher computational complexity

## Deployment Steps
1. **Pre-deployment**: Statically estimate layer sizes or dynamic profiling
2. **Partitioning**: Apply greedy or DP algorithm to determine boundaries
3. **Loading**: Load weights and pre-allocate memory within SRAM/L2 cache
4. **Execution**: Sequential layer execution on assigned card
5. **Communication**: Transfer intermediate outputs between partitions only when crossing device boundaries

## Edge Case Handling
- Single layer exceeding cache: apply intra-layer partitioning or compression
- Batch size tuning to reduce activation memory
- Variable layer sizes: adjust partitioning heuristics for memory utilization

## Model Specifications for Deployment
- Dense model: 16 layers
- Precision: FP16 (2 bytes per parameter)
- Batch size: 1024
- Architecture details:
  - Attention heads: 16
  - Head dimension: 512
  - MLP hidden size: 32768
  - Hidden size = 16 × 512 = 8192