# Phase 2: Methodology Extraction

## Problem Formulation
Given model with n layers L = {l₁, l₂, ..., lₙ}, partition into k disjoint groups P = {P₁, P₂, ..., Pₖ} such that:
- Memory footprint S(Pᵢ) ≤ C (SRAM/L2 cache capacity)
- Full execution order preserved (contiguous layer assignment)
- Number of partitions k minimized or balanced

Formula: S(Pᵢ) = Σ_{lⱼ∈Pᵢ} size(lⱼ) ≤ C

## Memory Footprint Estimation
Each layer size includes:
- **Weights**: Parameter tensors (datatype size × parameters)
- **Activations**: Intermediate outputs (output feature map dimensions × batch size)
- **Temporary Buffers**: Workspace memory for operators

Calculation: size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)

## Partitioning Algorithms

### 3.1 Greedy Layer Aggregation
1. Start from layer l₁
2. Initialize empty partition Pᵢ
3. Iteratively add layers lⱼ to Pᵢ, accumulating S(Pᵢ)
4. If adding lⱼ causes S(Pᵢ) > C, finalize Pᵢ with layers {l_start, ..., lⱼ₋₁}
5. Start new partition Pᵢ₊₁ from layer lⱼ
6. Repeat until all layers assigned

### 3.2 Dynamic Programming (Optional)
- Optimizes partition boundaries to minimize maximum partition size
- Achieves more balanced load while respecting cache capacity

## Deployment Strategy
1. After partitioning, assign each group Pᵢ to separate accelerator card
2. Load all weights and pre-allocate activation/buffer memory within SRAM/L2 cache
3. Execute layers sequentially on assigned card
4. Transfer intermediate outputs only between partitions on different cards

## Edge Cases Handling
- Single layer exceeding capacity C: use intra-layer partitioning or model compression
- Batch size tuning to reduce activation memory
- Variable layer sizes: adjust partitioning heuristics to avoid under-utilization

## Advantages
- Reduced memory access latency (minimizes off-chip DRAM access)
- Improved throughput (faster memory access + parallel execution)
- Scalability (adaptable to varying model sizes and hardware configurations)