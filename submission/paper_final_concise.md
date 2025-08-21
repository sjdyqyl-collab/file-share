# Layer-wise Deployment Strategy for Large Neural Networks

## Abstract
In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## Problem and Solution Overview

**Challenge**: Large neural networks face memory access bottlenecks due to limited on-chip memory (SRAM/L2 cache), with off-chip access introducing significant latency.

**Solution**: Layer-wise partitioning that splits *n* layers into *k* disjoint groups, each fitting within cache capacity *C*, while preserving layer execution order.

## Methodology

### 1. Problem Formulation
Given layers $L = {l_1, l_2, ..., l_n}$, partition into $k$ groups $P = {P_1, P_2, ..., P_k}$ such that:
- Each partition fits in SRAM/L2 cache: $S(P_i) = \sum_{l_j \in P_i} \text{size}(l_j) \leq C$
- Layers assigned contiguously in original order
- Minimize number of partitions $k$

### 2. Memory Footprint Estimation
For each layer $l_j$:
$$\text{size}(l_j) = \text{weight_size}(l_j) + \text{activation_size}(l_j) + \text{buffer_size}(l_j)$$

Components:
- **Weights**: Parameters × datatype size (FP16 = 2 bytes)
- **Activations**: Output feature maps × batch size
- **Buffers**: Operator workspace requirements

### 3. Partitioning Algorithms
- **Greedy approach**: Sequentially add layers to partition until cache limit reached
- **Dynamic programming**: Optimize for balanced partitions (optional)

### 4. Deployment Strategy
1. Load each partition entirely into on-chip memory
2. Execute layers sequentially on assigned card
3. Transfer data only between partitions on different cards

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Models**: 
  - Dense: 16-layer fully connected network
  - MoE: 16-layer mixture-of-experts with 8 experts/layer
- **Configuration**: FP16 precision, batch size 1024, 16 heads × 512 dim, MLP hidden size 32768
- **Baseline**: Tensor parallelism (TP=8) + Pipeline parallelism (PP=2)

### Results
| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense | Baseline | 16 | 12,800 | 0.078 |
| Dense | Proposed | 16 | 15,360 | 0.065 |
| MoE | Baseline | 16 | 10,200 | 0.098 |
| MoE | Proposed | 16 | 13,400 | 0.075 |

### Performance Gains
- **Dense model**: 20% TPS increase, 17% TPOT reduction
- **MoE model**: 31% TPS increase, 23% TPOT reduction

## Conclusion
Our layer-wise deployment strategy achieves substantial performance gains by explicitly considering on-chip memory constraints, with up to 31% throughput improvement over traditional parallelism approaches. The method is scalable and adaptable to varying model sizes and hardware configurations.