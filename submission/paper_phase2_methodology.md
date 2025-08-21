# Layer-wise Deployment Strategy - Methodology

## Abstract
In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## Methodology

### 1. Problem Formulation

Given a large model composed of *n* layers $L = {l_1, l_2, ..., l_n}$, the goal is to partition these layers into $k$ disjoint groups $P = {P_1, P_2, ..., P_k}$, each assigned to a separate hardware accelerator card, such that:

* The memory footprint of each group $P_i$ does not exceed the capacity $C$ of the SRAM or L2 cache available on the corresponding card.
* The full execution order of the model is preserved, i.e., layers are assigned contiguously in the original order.
* The number of partitions $k$ is minimized or balanced to maximize hardware utilization.

Formally, for each partition $P_i$, the size $S(P_i)$ satisfies:

$$
S(P_i) = sum_{l_j in P_i} text{size}(l_j) leq C
$$

where $text{size}(l_j)$ is the estimated memory footprint of layer $l_j$.

### 2. Memory Footprint Estimation

The memory footprint of each layer includes:

* **Weights**: The parameter tensors stored for the layer.
* **Activations**: Intermediate outputs needed during inference or training.
* **Temporary Buffers**: Workspace memory required by operators during computation.

To accurately estimate $text{size}(l_j)$, we calculate:

$$
text{size}(l_j) = text{weight_size}(l_j) + text{activation_size}(l_j) + text{buffer_size}(l_j)
$$

* **Weight size** is computed based on the number of parameters and their datatype size (e.g., FP16 = 2 bytes).
* **Activation size** depends on the output feature map dimensions and batch size.
* **Buffer size** is derived from profiling or analytical models of operator requirements.

This estimation can be performed statically prior to deployment or dynamically profiled for accuracy.

### 3. Partitioning Algorithm

Our method applies a greedy or dynamic programming algorithm to determine layer partitions:

#### 3.1 Greedy Layer Aggregation

Starting from the first layer $l_1$:

1. Initialize an empty partition $P_i$.
2. Iteratively add subsequent layers $l_j$ to $P_i$, accumulating $S(P_i)$.
3. If adding $l_j$ causes $S(P_i) > C$, finalize $P_i$ with layers ${l_{start}, ..., l_{j-1}}$.
4. Start a new partition $P_{i+1}$ beginning from layer $l_j$.
5. Repeat until all layers are assigned.

This approach is simple and efficient, guaranteeing that each partition fits the cache.

#### 3.2 Dynamic Programming for Balanced Partitions (Optional)

To achieve more balanced load and minimize the number of partitions, a dynamic programming (DP) approach can be employed to optimize partition boundaries. The DP algorithm tries to minimize the maximum partition size while respecting the cache capacity constraint.

### 4. Deployment Strategy

After partitioning, each group $P_i$ is deployed on a separate accelerator card with the following steps:

* Load all weights and pre-allocate activation and buffer memory within the SRAM or L2 cache.
* Execute the layers sequentially on the assigned card.
* Transfer intermediate outputs only when passing data between partitions on different cards, minimizing inter-card communication.

### 5. Handling Edge Cases

* If a single layer's memory footprint exceeds $C$, further intra-layer partitioning or model compression techniques (e.g., quantization, pruning) may be necessary.
* Batch size tuning can help reduce activation memory footprint to fit constraints.
* For models with highly variable layer sizes, partitioning heuristics can be adjusted to avoid under-utilization of on-chip memory.

### 6. Summary of Advantages

* **Reduced Memory Access Latency**: By fitting partitions in SRAM/L2 cache, off-chip DRAM accesses are minimized.
* **Improved Throughput**: Faster memory access and parallel execution on multiple cards increase inference/training speed.
* **Scalability**: The method can be adapted to varying model sizes and hardware configurations.