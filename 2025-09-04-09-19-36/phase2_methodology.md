# Phase 2: Methodology Extraction

## Problem Formulation
Given a large model composed of *n* layers $L = {l_1, l_2, ..., l_n}$, the goal is to partition these layers into $k$ disjoint groups $P = {P_1, P_2, ..., P_k}$, each assigned to a separate hardware accelerator card, such that:
- The memory footprint of each group $P_i$ does not exceed the capacity $C$ of the SRAM or L2 cache available on the corresponding card
- The full execution order of the model is preserved, i.e., layers are assigned contiguously in the original order
- The number of partitions $k$ is minimized or balanced to maximize hardware utilization

Formally, for each partition $P_i$, the size $S(P_i)$ satisfies:
$$S(P_i) = \sum_{l_j \in P_i} \text{size}(l_j) \leq C$$
where $\text{size}(l_j)$ is the estimated memory footprint of layer $l_j$.

## Memory Footprint Estimation
The memory footprint of each layer includes:
- **Weights**: The parameter tensors stored for the layer
- **Activations**: Intermediate outputs needed during inference or training
- **Temporary Buffers**: Workspace memory required by operators during computation

Calculation formula:
$$\text{size}(l_j) = \text{weight_size}(l_j) + \text{activation_size}(l_j) + \text{buffer_size}(l_j)$$

Where:
- **Weight size** is computed based on the number of parameters and their datatype size (e.g., FP16 = 2 bytes)
- **Activation size** depends on the output feature map dimensions and batch size
- **Buffer size** is derived from profiling or analytical models of operator requirements

## Partitioning Algorithms

### 3.1 Greedy Layer Aggregation
Starting from the first layer $l_1$:
1. Initialize an empty partition $P_i$
2. Iteratively add subsequent layers $l_j$ to $P_i$, accumulating $S(P_i)$
3. If adding $l_j$ causes $S(P_i) > C$, finalize $P_i$ with layers ${l_{start}, ..., l_{j-1}}$
4. Start a new partition $P_{i+1}$ beginning from layer $l_j$
5. Repeat until all layers are assigned

This approach is simple and efficient, guaranteeing that each partition fits the cache.

### 3.2 Dynamic Programming for Balanced Partitions (Optional)
To achieve more balanced load and minimize the number of partitions, a dynamic programming (DP) approach can be employed to optimize partition boundaries. The DP algorithm tries to minimize the maximum partition size while respecting the cache capacity constraint.

## Deployment Strategy
After partitioning, each group $P_i$ is deployed on a separate accelerator card with the following steps:
1. Load all weights and pre-allocate activation and buffer memory within the SRAM or L2 cache
2. Execute the layers sequentially on the assigned card
3. Transfer intermediate outputs only when passing data between partitions on different cards, minimizing inter-card communication

## Handling Edge Cases
- If a single layer's memory footprint exceeds $C$, further intra-layer partitioning or model compression techniques (e.g., quantization, pruning) may be necessary
- Batch size tuning can help reduce activation memory footprint to fit constraints
- For models with highly variable layer sizes, partitioning heuristics can be adjusted to avoid under-utilization of on-chip memory

## Model Specifications for Experiments
- Dense model: 16-layer fully connected dense network
- Precision: FP16 (2 bytes per parameter)
- Batch size: 1024
- Number of heads: 16
- Dimension per head: 512
- Hidden size of MLP: 32768