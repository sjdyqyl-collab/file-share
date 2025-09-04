# Concise Paper: Layer-wise Deployment Strategy for Large Neural Networks

### Abstract

In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## Introduction

The rapid growth of deep learning model sizes has posed significant challenges for efficient deployment, especially on hardware with limited on-chip memory such as SRAM and L2 cache. Large models typically require access to external memory, which introduces latency and bandwidth bottlenecks that degrade overall system performance. To address this, it is crucial to design deployment strategies that leverage the fast but limited on-chip memory available in modern accelerators.

## Methodology

### Problem Formulation
Given a large model composed of *n* layers $L = {l_1, l_2, ..., l_n}$, partition these layers into $k$ disjoint groups $P = {P_1, P_2, ..., P_k}$, each assigned to a separate hardware accelerator card, such that:
- Memory footprint of each group $P_i$ does not exceed cache capacity $C$
- Full execution order is preserved (contiguous layers)
- Number of partitions $k$ is minimized

$$S(P_i) = \sum_{l_j \in P_i} \text{size}(l_j) \leq C$$

### Memory Footprint Estimation
$$\text{size}(l_j) = \text{weight_size}(l_j) + \text{activation_size}(l_j) + \text{buffer_size}(l_j)$$

Where:
- **Weight size**: Parameters × datatype (FP16 = 2 bytes)
- **Activation size**: Output feature map × batch size
- **Buffer size**: Operator workspace requirements

### Partitioning Algorithms

#### Greedy Layer Aggregation
1. Initialize empty partition $P_i$
2. Iteratively add layers $l_j$ to $P_i$, accumulating $S(P_i)$
3. If $S(P_i) > C$, finalize $P_i$ with layers ${l_{start}, ..., l_{j-1}}$
4. Start new partition $P_{i+1}$ from layer $l_j$
5. Repeat until all layers assigned

#### Dynamic Programming (Optional)
Optimizes partition boundaries to minimize maximum partition size while respecting cache constraints.

### Deployment Strategy
1. Load weights and pre-allocate memory within SRAM/L2 cache
2. Execute layers sequentially on assigned card
3. Transfer intermediate outputs only between partitions

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Model**: 16-layer dense network
- **Precision**: FP16
- **Batch size**: 1024
- **Dimensions**: 16 heads, 512 dim/head, 32768 MLP hidden size
- **Baseline**: TP=8, PP=2 (16 GPUs total)

### Results
| Model | Method | GPUs | TPS | TPOT |
|-------|--------|------|-----|------|
| Dense | Baseline | 16 | 12,800 | 0.078ms |
| Dense | Proposed | 16 | 15,360 | 0.065ms |

**Improvements**: 20% TPS increase, 17% TPOT reduction

## Conclusion

The layer-wise deployment strategy achieves substantial performance gains by explicitly considering on-chip memory constraints during layer partitioning. Experimental results demonstrate up to 20% improvement in throughput and corresponding latency reduction compared to standard tensor and pipeline parallelism approaches.