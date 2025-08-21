# Layer-wise Deployment Strategy for Large Neural Networks

### Abstract

In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## 1. Introduction

Large neural networks face deployment challenges due to limited on-chip memory (SRAM/L2 cache) and off-chip memory access bottlenecks. We propose layer-wise partitioning that splits *n* layers into *k* groups across multiple accelerator cards, ensuring each partition fits within SRAM/L2 cache capacity *C* while preserving execution order and minimizing partitions.

## 2. Methodology

### 2.1 Problem Formulation
Given *n* layers $L = {l_1, l_2, ..., l_n}$, partition into $k$ disjoint groups $P = {P_1, P_2, ..., P_k}$ such that:
- Each $P_i$ assigned to separate accelerator card
- Memory footprint $S(P_i) \leq C$ (cache capacity)
- Contiguous layer assignment preserving execution order
- Minimize number of partitions $k$

**Constraint:**
$$S(P_i) = \sum_{l_j \in P_i} \text{size}(l_j) \leq C$$

### 2.2 Memory Footprint Estimation
For each layer $l_j$:
$$\text{size}(l_j) = \text{weight_size}(l_j) + \text{activation_size}(l_j) + \text{buffer_size}(l_j)$$

Where:
- **weight_size**: Parameters × datatype (FP16=2 bytes)
- **activation_size**: Output dimensions × batch size
- **buffer_size**: Operator workspace requirements

### 2.3 Partitioning Algorithms

#### Greedy Layer Aggregation
1. Initialize empty partition $P_i$
2. Iteratively add layers $l_j$ to $P_i$, accumulating $S(P_i)$
3. If $S(P_i) > C$, finalize $P_i$ with layers ${l_{start}, ..., l_{j-1}}$
4. Start new partition $P_{i+1}$ from layer $l_j$
5. Repeat until all layers assigned

#### Dynamic Programming (Optional)
- Minimize maximum partition size while respecting cache capacity
- Achieves more balanced load distribution

### 2.4 Deployment Strategy
1. **Load phase**: Load weights and pre-allocate memory within SRAM/L2 cache
2. **Execution phase**: Execute layers sequentially on assigned card
3. **Communication**: Transfer outputs only between partitions on different cards

### 2.5 Edge Case Handling
- **Oversized layers**: Apply intra-layer partitioning or compression
- **Variable layer sizes**: Adjust partitioning heuristics
- **Batch size tuning**: Reduce activation memory footprint

## 3. Experiments

### 3.1 Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Models**: 
  - Dense: 16-layer fully connected network
  - MoE: 16-layer mixture-of-experts (8 experts/layer)
- **Parameters**: FP16, batch size=1024, 16 heads, 512 head dim, 32768 MLP hidden size
- **Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- **Metrics**: TPS (tokens/second), TPOT (time/output token in ms)

### 3.2 Results

| Model | Method | GPUs | TPS | TPOT |
|-------|--------|------|-----|------|
| Dense | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078ms |
| Dense | Proposed Layer-wise | 16 | 15,360 | 0.065ms |
| MoE | Baseline (TP=8, PP=2) | 16 | 10,200 | 0.098ms |
| MoE | Proposed Layer-wise | 16 | 13,400 | 0.075ms |

### 3.3 Analysis
- **Dense model**: 20% TPS increase, 17% latency reduction
- **MoE model**: 31% TPS increase, 23% latency reduction
- Greater benefits for complex models due to reduced communication overhead
- Cache-aware deployment outperforms traditional TP+PP approaches

## 4. Conclusion

Our layer-wise deployment strategy explicitly partitions model layers across multiple accelerators with cache capacity constraints, achieving up to 31% throughput improvement and 23% latency reduction. The method enables scalable deployment of large models while maximizing fast memory utilization.

## DAG Generation Requirements

For experimental model deployment DAG:
1. **Input**: Model with *n* layers, cache capacity *C* per device
2. **Partitioning**: Apply greedy/dynamic programming to create *k* partitions
3. **Mapping**: Assign each partition $P_i$ to GPU $i$
4. **Execution**: Sequential layer execution within each partition
5. **Communication**: Transfer activations only between partition boundaries
6. **Memory**: Ensure $S(P_i) \leq C$ for all partitions