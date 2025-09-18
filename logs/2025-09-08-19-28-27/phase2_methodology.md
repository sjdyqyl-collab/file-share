# Phase 2: Methodology Extraction

## Problem Formulation
Given a large model composed of *n* layers $L = {l_1, l_2, ..., l_n}$, partition these layers into $k$ disjoint groups $P = {P_1, P_2, ..., P_k}$, each assigned to a separate hardware accelerator card, such that:
- The memory footprint of each group $P_i$ does not exceed the capacity $C$ of the SRAM or L2 cache available on the corresponding card
- The full execution order of the model is preserved (layers assigned contiguously in original order)
- The number of partitions $k$ is minimized or balanced to maximize hardware utilization

For each partition $P_i$, the size $S(P_i)$ satisfies:
$$
S(P_i) = \sum_{l_j \in P_i} \text{size}(l_j) \leq C
$$

## Memory Footprint Estimation
The memory footprint of each layer includes:
- **Weights**: Parameter tensors stored for the layer
- **Activations**: Intermediate outputs needed during inference or training
- **Temporary Buffers**: Workspace memory required by operators during computation

Calculation formula:
$$
\text{size}(l_j) = \text{weight_size}(l_j) + \text{activation_size}(l_j) + \text{buffer_size}(l_j)
$$

Where:
- **Weight size** = number of parameters × datatype size (FP16 = 2 bytes)
- **Activation size** = output feature map dimensions × batch size
- **Buffer size** = derived from profiling or analytical models of operator requirements

## Partitioning Algorithms

### 1. Greedy Layer Aggregation
Starting from the first layer $l_1$:
1. Initialize an empty partition $P_i$
2. Iteratively add subsequent layers $l_j$ to $P_i$, accumulating $S(P_i)$
3. If adding $l_j$ causes $S(P_i) > C$, finalize $P_i$ with layers ${l_{start}, ..., l_{j-1}}$
4. Start a new partition $P_{i+1}$ beginning from layer $l_j$
5. Repeat until all layers are assigned

This approach guarantees each partition fits the cache and is simple to implement.

### 2. Dynamic Programming for Balanced Partitions (Optional)
To achieve more balanced load and minimize the number of partitions, a dynamic programming approach optimizes partition boundaries by minimizing the maximum partition size while respecting the cache capacity constraint.

## Deployment Strategy
After partitioning, each group $P_i$ is deployed on a separate accelerator card:
1. Load all weights and pre-allocate activation and buffer memory within the SRAM or L2 cache
2. Execute the layers sequentially on the assigned card
3. Transfer intermediate outputs only when passing data between partitions on different cards, minimizing inter-card communication

## Edge Case Handling
- **Single layer exceeds C**: Apply intra-layer partitioning or model compression (quantization, pruning)
- **Large activation footprint**: Tune batch size to reduce activation memory
- **Variable layer sizes**: Adjust partitioning heuristics to avoid under-utilization of on-chip memory

## Model Specifications for Experiments
- **Dense model**: 16-layer fully connected dense network
- **Precision**: FP16 (2 bytes per parameter)
- **Batch size**: 1024
- **Head configuration**: 16 heads, 512 dimensions per head
- **MLP hidden size**: 32768
- **Hardware**: 16 NVIDIA H100 GPUs