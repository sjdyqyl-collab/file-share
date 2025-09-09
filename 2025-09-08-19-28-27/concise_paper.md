# Layer-Wise Deployment Strategy for Large Neural Network Models

### Abstract
In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## 1. Introduction
Large neural network models face deployment challenges due to memory hierarchy constraints. While accelerators offer high computational throughput, their on-chip memory (SRAM/L2 cache) is limited compared to external DRAM. Accessing on-chip memory is significantly faster and more energy-efficient than off-chip memory.

This paper introduces a layer-wise partitioning method that splits *n* model layers across *k* accelerator cards, ensuring each partition fits entirely within the on-chip memory capacity *C* of the target hardware.

## 2. Methodology

### 2.1 Problem Formulation
Given *n* layers $L = {l_1, l_2, ..., l_n}$, partition into *k* disjoint groups $P = {P_1, P_2, ..., P_k}$ such that:
- Memory footprint $S(P_i) \leq C$ (cache capacity)
- Preserve layer execution order (contiguous assignment)
- Minimize *k* for balanced hardware utilization

### 2.2 Memory Footprint Estimation
For each layer $l_j$:
$$
\text{size}(l_j) = \text{weight_size}(l_j) + \text{activation_size}(l_j) + \text{buffer_size}(l_j)
$$
- **Weight size**: Parameters × datatype (FP16 = 2 bytes)
- **Activation size**: Output dimensions × batch size
- **Buffer size**: Operator workspace requirements

### 2.3 Partitioning Algorithms

#### Greedy Layer Aggregation
1. Start from layer $l_1$, initialize empty partition $P_i$
2. Add layers sequentially until $S(P_i) > C$
3. Finalize partition, start new partition from current layer
4. Guarantee: Each partition fits cache capacity

#### Dynamic Programming (Optional)
Optimize partition boundaries to minimize maximum partition size while respecting cache constraints.

### 2.4 Deployment Strategy
- Load each partition $P_i$ into SRAM/L2 cache of separate accelerator card
- Execute layers sequentially on assigned card
- Transfer intermediate outputs only between partitions on different cards

## 3. Experiments

### 3.1 Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Model**: 16-layer dense network
- **Precision**: FP16
- **Batch size**: 1024
- **Architecture**: 16 heads, 512 dims/head, MLP hidden size 32768

### 3.2 Baseline
- **Method**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- **Utilization**: 8 × 2 = 16 GPUs

### 3.3 Results

| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

### 3.4 Analysis
- **20% TPS improvement** (12,800 → 15,360)
- **17% TPOT reduction** (0.078ms → 0.065ms)
- Root cause: Efficient on-chip memory utilization, reduced off-chip accesses

## 4. Conclusion
The proposed layer-wise deployment strategy achieves significant performance gains by explicitly considering on-chip memory constraints during layer partitioning. Results demonstrate up to 20% throughput improvement over traditional tensor and pipeline parallelism approaches, validating the effectiveness of cache-aware deployment for large-scale neural networks.