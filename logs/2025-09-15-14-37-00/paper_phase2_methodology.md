# Methodology - Phase 2

## 1. Problem Formulation

Given a large model composed of *n* layers $L = {l_1, l_2, ..., l_n}$, partition these layers into $k$ disjoint groups $P = {P_1, P_2, ..., P_k}$, each assigned to a separate hardware accelerator card, such that:

- The memory footprint of each group $P_i$ does not exceed the capacity $C$ of the SRAM or L2 cache available on the corresponding card
- The full execution order of the model is preserved (layers assigned contiguously in original order)
- The number of partitions $k$ is minimized or balanced to maximize hardware utilization

**Formal constraint:**
$$
S(P_i) = \sum_{l_j \in P_i} \text{size}(l_j) \leq C
$$

## 2. Memory Footprint Estimation

### 2.1 Components
Each layer's memory footprint includes:
- **Weights**: Parameter tensors stored for the layer
- **Activations**: Intermediate outputs needed during inference/training
- **Temporary Buffers**: Workspace memory required by operators during computation

### 2.2 Calculation Formula
$$
\text{size}(l_j) = \text{weight_size}(l_j) + \text{activation_size}(l_j) + \text{buffer_size}(l_j)
$$

### 2.3 Detailed Specifications
- **Weight size**: Number of parameters × datatype size (FP16 = 2 bytes)
- **Activation size**: Output feature map dimensions × batch size
- **Buffer size**: Derived from profiling or analytical models of operator requirements

## 3. Partitioning Algorithms

### 3.1 Greedy Layer Aggregation Algorithm

**Algorithm Steps:**
1. Initialize empty partition $P_i$
2. Starting from first layer $l_1$, iteratively add subsequent layers $l_j$ to $P_i$
3. Accumulate $S(P_i)$ with each layer addition
4. If adding $l_j$ causes $S(P_i) > C$, finalize $P_i$ with layers $\{l_{start}, ..., l_{j-1}\}$
5. Start new partition $P_{i+1}$ beginning from layer $l_j$
6. Repeat until all layers assigned

**Properties:**
- Simple and efficient
- Guarantees each partition fits cache
- May create imbalanced partitions

### 3.2 Dynamic Programming for Balanced Partitions

**Objective:** Minimize maximum partition size while respecting cache capacity constraint
- Uses DP algorithm to optimize partition boundaries
- Achieves more balanced load distribution
- Minimizes number of partitions $k$

## 4. Deployment Strategy

### 4.1 Pre-deployment Steps
1. **Memory estimation**: Calculate size for each layer using formula above
2. **Partitioning**: Apply greedy or DP algorithm to determine layer groups
3. **Validation**: Ensure each partition size ≤ cache capacity $C$

### 4.2 Runtime Deployment
For each partition $P_i$ on accelerator card $i$:
1. **Load phase**: Load all weights and pre-allocate activation/buffer memory within SRAM/L2 cache
2. **Execution**: Execute layers sequentially on assigned card
3. **Communication**: Transfer intermediate outputs only when passing data between partitions on different cards

### 4.3 Memory Layout Optimization
- **Contiguous allocation**: Ensure weights, activations, and buffers are contiguous in memory
- **Cache alignment**: Align memory allocations to cache line boundaries
- **Pre-allocation**: Allocate all required memory upfront to avoid runtime allocation overhead

## 5. Handling Edge Cases

### 5.1 Single Layer Exceeding Cache Capacity
- **Solution 1**: Intra-layer partitioning (split single layer across multiple cards)
- **Solution 2**: Model compression techniques (quantization, pruning)
- **Solution 3**: Reduce batch size to decrease activation memory

### 5.2 Highly Variable Layer Sizes
- **Solution**: Adjust partitioning heuristics to avoid under-utilization
- **Strategy**: Use sliding window approach to find optimal partition boundaries

### 5.3 Dynamic Workload Adaptation
- **Solution**: Runtime profiling to adjust partitions based on actual memory usage
- **Strategy**: Implement feedback loop for partition refinement

## 6. Implementation Parameters

### 6.1 Model Parameters
- **Dense model**: 16 layers
- **Precision**: FP16 (2 bytes per parameter)
- **Batch size**: 1024
- **Sequence length**: 10000
- **Heads**: 16
- **Head dimension**: 512
- **MLP hidden size**: 32768

### 6.2 Hardware Parameters
- **Devices**: 16 NVIDIA H100 GPUs
- **Cache capacity**: SRAM/L2 cache (exact capacity depends on device)
- **Interconnect**: High-speed interconnect between GPUs for partition communication

### 6.3 Baseline Comparison
- **Configuration**: TP=8, PP=2 (16 GPUs total)
- **Method**: Standard tensor parallelism + pipeline parallelism
- **Limitation**: Does not consider on-chip memory constraints explicitly