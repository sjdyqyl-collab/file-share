# Phase 2: Methodology Extraction

## Problem Formulation

Given a large model composed of *n* layers L = {l₁, l₂, ..., lₙ}, partition these layers into k disjoint groups P = {P₁, P₂, ..., Pₖ} with the following constraints:

### Mathematical Formulation
- Each partition Pᵢ assigned to a separate hardware accelerator card
- Memory footprint constraint: S(Pᵢ) ≤ C (cache capacity)
- Full execution order preservation: layers assigned contiguously in original order
- Optimization goal: minimize k or balance load

### Memory Size Calculation
For each partition Pᵢ:
```
S(Pᵢ) = Σ(lⱼ∈Pᵢ) size(lⱼ) ≤ C
```

## Memory Footprint Estimation

### Layer Memory Components
1. **weight_size(lⱼ)** = number_of_parameters × datatype_size
   - FP16: 2 bytes per parameter
   - FP32: 4 bytes per parameter

2. **activation_size(lⱼ)** = output_feature_map_dimensions × batch_size × datatype_size
   - Depends on layer output shape and batch size
   - For transformer layers: sequence_length × hidden_size × batch_size × datatype_size

3. **buffer_size(lⱼ)** = workspace_memory_for_operators
   - Derived from profiling or analytical models
   - Includes temporary storage for matrix operations

### Total Layer Size
```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
```

## Partitioning Algorithms

### 3.1 Greedy Layer Aggregation Algorithm

**Input**: Layers L = {l₁, l₂, ..., lₙ}, cache capacity C
**Output**: Partitions P = {P₁, P₂, ..., Pₖ}

**Algorithm Steps**:
1. Initialize i = 1, start = 1
2. While start ≤ n:
   - Initialize empty partition Pᵢ
   - Initialize accumulated_size = 0
   - For j from start to n:
     - If accumulated_size + size(lⱼ) ≤ C:
       - Add lⱼ to Pᵢ
       - accumulated_size += size(lⱼ)
     - Else:
       - Finalize Pᵢ with layers {l_start, ..., l_{j-1}}
       - Set start = j
       - Increment i
       - Break inner loop
   - If j == n and accumulated_size + size(lⱼ) ≤ C:
     - Add lⱼ to Pᵢ
     - Finalize Pᵢ

**Properties**:
- Guarantees each partition fits cache
- Simple O(n) complexity
- May create unbalanced partitions

### 3.2 Dynamic Programming for Balanced Partitions

**Objective**: Minimize maximum partition size while respecting cache constraint

**DP Formulation**:
- Let dp[i][j] = minimum maximum partition size for first i layers using j partitions
- Recurrence relation:
  ```
  dp[i][j] = min_{k<i} max(dp[k][j-1], sum_{m=k+1}^i size(lₘ))
  ```
- Constraint: sum_{m=k+1}^i size(lₘ) ≤ C

**Output**: Optimal partition boundaries minimizing load imbalance

## Deployment Strategy

### Step-by-Step Process

1. **Pre-deployment Analysis**
   - Estimate size(lⱼ) for each layer
   - Determine cache capacity C for target hardware
   - Run partitioning algorithm to get P = {P₁, P₂, ..., Pₖ}

2. **Resource Allocation**
   - Assign partition Pᵢ to accelerator card i
   - Ensure k ≤ available_cards

3. **Memory Loading**
   - Load all weights of Pᵢ into SRAM/L2 cache
   - Pre-allocate activation and buffer memory
   - Verify total memory ≤ C

4. **Execution Flow**
   - Card i executes layers in Pᵢ sequentially
   - Transfer output activations to card i+1 via high-speed interconnect
   - Minimal inter-card communication (only between partitions)

### Edge Case Handling

**Single Layer Exceeds Cache**:
- Apply intra-layer partitioning
- Use model compression (quantization, pruning)
- Reduce batch size to decrease activation memory

**Variable Layer Sizes**:
- Adjust partitioning heuristics
- Consider layer fusion for small layers
- Balance computation vs memory trade-offs

### Hardware-Specific Parameters

**NVIDIA H100 Example**:
- L2 cache: 50MB per GPU
- SRAM: Varies by implementation
- Interconnect: NVLink for high-speed transfers
- Memory bandwidth: Critical for inter-partition transfers

## Summary of Methodology

1. **Input**: Model with n layers, hardware with cache capacity C
2. **Process**: 
   - Estimate layer sizes
   - Apply partitioning algorithm
   - Map partitions to devices
3. **Output**: k partitions fitting cache constraints
4. **Execution**: Parallel processing with minimal inter-device communication