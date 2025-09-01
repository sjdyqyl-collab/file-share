# Layer-Wise Deployment Strategy for Large Neural Networks: A Concise Version

### Abstract

In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

---

## Introduction

The rapid growth of deep learning model sizes has posed significant challenges for efficient deployment, especially on hardware with limited on-chip memory such as SRAM and L2 cache. Large models typically require access to external memory, which introduces latency and bandwidth bottlenecks that degrade overall system performance. To address this, it is crucial to design deployment strategies that leverage the fast but limited on-chip memory available in modern accelerators.

This paper introduces a layer-wise partitioning and distribution method for large models, where *n* layers of the model are split and mapped onto multiple accelerator cards. The key objective is to ensure that each layer group assigned to a card can be entirely loaded into its SRAM or L2 cache. By doing so, we minimize memory access overhead and improve throughput during inference or training.

---

## Methodology

### Problem Formulation

Given a large model composed of *n* layers $L = {l_1, l_2, ..., l_n}$, partition these layers into $k$ disjoint groups $P = {P_1, P_2, ..., P_k}$, each assigned to a separate hardware accelerator card, such that:

* The memory footprint of each group $P_i$ does not exceed the capacity $C$ of the SRAM or L2 cache
* The full execution order of the model is preserved (contiguous assignment)
* The number of partitions $k$ is minimized or balanced to maximize hardware utilization

Formally: $S(P_i) = \sum_{l_j \in P_i} \text{size}(l_j) \leq C$

### Memory Footprint Estimation

Each layer's memory footprint includes:
* **Weights**: Parameter tensors (datatype size: FP16 = 2 bytes)
* **Activations**: Intermediate outputs (depends on output dimensions and batch size)
* **Temporary Buffers**: Workspace memory for operators

Calculation: $\text{size}(l_j) = \text{weight_size}(l_j) + \text{activation_size}(l_j) + \text{buffer_size}(l_j)$

### Partitioning Algorithms

**Greedy Layer Aggregation:**
1. Start from first layer $l_1$
2. Iteratively add layers to partition $P_i$ until $S(P_i) > C$
3. Finalize $P_i$ and start new partition $P_{i+1}$ from current layer
4. Repeat until all layers assigned

**Dynamic Programming (Optional):**
- Minimize maximum partition size while respecting cache capacity
- Achieve more balanced load distribution

### Deployment Strategy

1. **Per Card**: Load weights and pre-allocate activations/buffers within SRAM/L2 cache
2. **Execution**: Process layers sequentially on assigned card
3. **Communication**: Transfer data only between partitions on different cards

---

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Models**: 
  - Dense: 16-layer fully connected network
  - MoE: 16-layer mixture-of-experts with 8 experts per layer
- **Parameters**: FP16 precision, batch size 1024, 16 heads, head dimension 512, MLP hidden size 32768
- **Baseline**: TP=8, PP=2 (standard tensor and pipeline parallelism)

### Results

| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense | Proposed Layer-wise | 16 | 15,360 | 0.065 |
| MoE | Baseline (TP=8, PP=2) | 16 | 10,200 | 0.098 |
| MoE | Proposed Layer-wise | 16 | 13,400 | 0.075 |

### Performance Analysis
- **Dense Model**: 20% TPS increase, 17% TPOT reduction
- **MoE Model**: 31% TPS increase, 23% TPOT reduction
- **Key Advantage**: Explicit on-chip memory optimization reduces off-chip accesses and communication delays

---

## Conclusion

We proposed a layer-wise deployment strategy that partitions model layers across multiple accelerator cards with the constraint that each partition fits entirely within SRAM or L2 cache. This approach significantly reduces off-chip memory accesses and improves inference efficiency. Experimental results demonstrate substantial performance gains over baseline tensor and pipeline parallelism, with up to 31% improvement in throughput and corresponding latency reduction.

---

## Key Technical Specifications

- **Layer Count**: 16 layers (both dense and MoE models)
- **Experts per Layer**: 8 (MoE model)
- **Precision**: FP16 (2 bytes per parameter)
- **Batch Size**: 1024
- **Architecture Details**: 16 heads, 512 head dimension, 32768 MLP hidden size
- **Hardware**: 16 NVIDIA H100 GPUs
- **Cache Constraint**: Each partition must fit within device SRAM/L2 cache capacity C