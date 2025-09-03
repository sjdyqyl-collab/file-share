# Phase 2: Methodology Extraction

## Methodology: Layer-wise Deployment Strategy

### 1. Problem Formulation
Given a large model composed of n layers L = {l₁, l₂, ..., lₙ}, partition these layers into k disjoint groups P = {P₁, P₂, ..., Pₖ}, each assigned to a separate hardware accelerator card, such that:

- The memory footprint of each group Pᵢ does not exceed the capacity C of the SRAM or L2 cache available on the corresponding card
- The full execution order of the model is preserved (layers assigned contiguously in original order)
- The number of partitions k is minimized or balanced to maximize hardware utilization

**Mathematical constraint:**
For each partition Pᵢ, the size S(Pᵢ) satisfies:
```
S(Pᵢ) = Σ_{lⱼ ∈ Pᵢ} size(lⱼ) ≤ C
```
where size(lⱼ) is the estimated memory footprint of layer lⱼ.

### 2. Memory Footprint Estimation
The memory footprint of each layer includes:
- **Weights**: Parameter tensors stored for the layer
- **Activations**: Intermediate outputs needed during inference/training
- **Temporary Buffers**: Workspace memory required by operators during computation

**Calculation formula:**
```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
```

**Component details:**
- **Weight size**: Number of parameters × datatype size (FP16 = 2 bytes)
- **Activation size**: Output feature map dimensions × batch size
- **Buffer size**: Derived from profiling or analytical models of operator requirements

### 3. Partitioning Algorithms

#### 3.1 Greedy Layer Aggregation
Starting from the first layer l₁:
1. Initialize an empty partition Pᵢ
2. Iteratively add subsequent layers lⱼ to Pᵢ, accumulating S(Pᵢ)
3. If adding lⱼ causes S(Pᵢ) > C, finalize Pᵢ with layers {l_start, ..., l_{j-1}}
4. Start a new partition P_{i+1} beginning from layer lⱼ
5. Repeat until all layers are assigned

**Properties:** Simple and efficient, guarantees each partition fits the cache

#### 3.2 Dynamic Programming for Balanced Partitions (Optional)
To achieve more balanced load and minimize the number of partitions, a dynamic programming approach optimizes partition boundaries by minimizing the maximum partition size while respecting cache capacity constraint.

### 4. Deployment Strategy
After partitioning, each group Pᵢ is deployed on a separate accelerator card:
1. Load all weights and pre-allocate activation and buffer memory within SRAM/L2 cache
2. Execute layers sequentially on the assigned card
3. Transfer intermediate outputs only when passing data between partitions on different cards, minimizing inter-card communication

### 5. Handling Edge Cases
- **Oversized single layer**: If a single layer's memory footprint exceeds C, apply intra-layer partitioning or model compression (quantization, pruning)
- **Activation constraints**: Batch size tuning can reduce activation memory footprint
- **Variable layer sizes**: Adjust partitioning heuristics to avoid under-utilization of on-chip memory

### 6. Advantages Summary
- **Reduced Memory Access Latency**: Minimize off-chip DRAM accesses by fitting partitions in SRAM/L2 cache
- **Improved Throughput**: Faster memory access and parallel execution on multiple cards
- **Scalability**: Adaptable to varying model sizes and hardware configurations