# Methodology - Phase 2

## Problem Formulation
Given a large model composed of *n* layers L = {l₁, l₂, ..., lₙ}, partition into *k* disjoint groups P = {P₁, P₂, ..., Pₖ}, each assigned to separate hardware accelerator card with constraints:
- Memory footprint S(Pᵢ) ≤ C (cache capacity)
- Full execution order preserved (contiguous assignment)
- Minimize or balance *k* for hardware utilization

## Memory Footprint Estimation
For each layer lⱼ:
```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
```

Where:
- **weight_size**: parameters × datatype size (FP16 = 2 bytes)
- **activation_size**: output feature map dimensions × batch size
- **buffer_size**: workspace memory for operators (profiled or analytical)

## Partitioning Algorithms

### 1. Greedy Layer Aggregation
```
1. Initialize empty partition Pᵢ
2. Iteratively add layers lⱼ to Pᵢ, accumulating S(Pᵢ)
3. If S(Pᵢ) > C: finalize Pᵢ with {l_start, ..., l_{j-1}}
4. Start new partition P_{i+1} from layer lⱼ
5. Repeat until all layers assigned
```

### 2. Dynamic Programming (Optional)
Optimizes partition boundaries to minimize maximum partition size while respecting cache capacity constraint.

## Deployment Strategy
1. Load all weights of partition Pᵢ into SRAM/L2 cache
2. Pre-allocate activation and buffer memory within cache
3. Execute layers sequentially on assigned card
4. Transfer intermediate outputs only between partitions on different cards

## Edge Case Handling
- Single layer exceeding cache capacity: apply intra-layer partitioning or model compression (quantization/pruning)
- Batch size tuning to reduce activation memory
- Variable layer sizes: adjust partitioning heuristics to avoid under-utilization

## Advantages
- Reduced memory access latency (minimized off-chip DRAM access)
- Improved throughput (faster memory access + parallel execution)
- Scalability (adaptable to varying model sizes and hardware)