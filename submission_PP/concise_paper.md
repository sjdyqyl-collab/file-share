# Layer-wise Deployment Strategy for Large Neural Networks

### Abstract
In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## Introduction
The rapid growth of deep learning model sizes has posed significant challenges for efficient deployment, especially on hardware with limited on-chip memory such as SRAM and L2 cache. Large models typically require access to external memory, which introduces latency and bandwidth bottlenecks that degrade overall system performance. To address this, it is crucial to design deployment strategies that leverage the fast but limited on-chip memory available in modern accelerators.

This paper introduces a layer-wise partitioning and distribution method for large models, where *n* layers of the model are split and mapped onto multiple accelerator cards. The key objective is to ensure that each layer group assigned to a card can be entirely loaded into its SRAM or L2 cache. By doing so, we minimize memory access overhead and improve throughput during inference or training.

## Methodology

### Problem Formulation
Given a large model composed of *n* layers $L = {l_1, l_2, ..., l_n}$, partition these layers into $k$ disjoint groups $P = {P_1, P_2, ..., P_k}$, each assigned to a separate hardware accelerator card, such that:
- Memory footprint of each group $P_i$ does not exceed cache capacity $C$
- Full execution order preserved with contiguous layer assignment
- Number of partitions $k$ is minimized

Formally: $S(P_i) = sum_{l_j in P_i} text{size}(l_j) leq C$

### Memory Footprint Estimation
$text{size}(l_j) = text{weight_size}(l_j) + text{activation_size}(l_j) + text{buffer_size}(l_j)$

- **Weights**: parameter tensors (FP16 = 2 bytes per parameter)
- **Activations**: intermediate outputs (output_feature_map × batch_size)
- **Buffers**: workspace memory from operator profiling

### Partitioning Algorithm

#### Greedy Layer Aggregation
1. Initialize empty partition $P_i$
2. Iteratively add layers while $S(P_i) leq C$
3. When capacity exceeded, finalize partition and start new one
4. Repeat until all layers assigned

#### Dynamic Programming (Optional)
For balanced partitions, use DP to minimize maximum partition size while respecting cache constraints.

### Deployment Strategy
- Load weights and pre-allocate memory in SRAM/L2 cache
- Execute layers sequentially on assigned GPU
- Transfer intermediate outputs only between partitions

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Models**: 
  - Dense: 16-layer fully connected network
  - MoE: 16-layer mixture-of-experts with 8 experts/layer
- **Configuration**: FP16 precision, batch_size=1024, 16 heads, 512 head_dim, 32768 MLP hidden_size
- **Baseline**: TP=8, PP=2 (standard tensor + pipeline parallelism)
- **Metrics**: Tokens Per Second (TPS), Time Per Output Token (TPOT)

### Results
| Model | Method | GPUs | TPS | TPOT |
|-------|--------|------|-----|------|
| Dense | Baseline | 16 | 12,800 | 0.078ms |
| Dense | Proposed | 16 | 15,360 | 0.065ms |
| MoE | Baseline | 16 | 10,200 | 0.098ms |
| MoE | Proposed | 16 | 13,400 | 0.075ms |

### Analysis
- **Dense model**: 20% TPS increase, 17% TPOT reduction
- **MoE model**: 31% TPS increase, 23% TPOT reduction
- Gains from efficient on-chip memory utilization and reduced communication overhead

## Conclusion
We proposed a layer-wise deployment strategy that partitions model layers across multiple accelerator cards with the constraint that each partition fits within SRAM/L2 cache. This approach significantly reduces off-chip memory accesses and improves inference efficiency. Experimental results demonstrate substantial performance gains over baseline tensor and pipeline parallelism, with up to 31% improvement in throughput and corresponding latency reduction.