# Helix: Two-Level Attention Partitioning for Large-Scale Transformers

## Abstract
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Introduction
Transformer architectures with multi-head attention (MHA) face scaling challenges when distributing computations across hardware units. Traditional MHA parallelization splits attention heads across devices but is limited by the fixed number of heads, leading to suboptimal utilization when the number of devices exceeds the number of heads. We introduce a two-level partitioning strategy that extends beyond conventional head-wise splitting by further segmenting each attention head's internal dimension.

## Method

### Two-Level Partitioning Scheme

#### Parameters
- $h$: total number of heads
- $d$: dimension per head ($D = h \times d$)
- $n$: number of head partitions
- $m$: number of dimension partitions per head
- $h_g = h/n$: heads per group
- $d_s = d/m$: slice dimension per partition

#### Mathematical Formulation
Given input $X \in \mathbb{R}^{B \times L \times D}$, the MHA layer projects:
$$Q, K, V = XW_Q, XW_K, XW_V$$

Each projection matrix $W \in \mathbb{R}^{D \times D}$ is partitioned into blocks $W^{(i,j)}$ where $i \in [1,n]$ indexes head groups and $j \in [1,m]$ indexes dimension slices.

Each device computes for partition $(i,j)$:
$$Q^{(i,j)} = X W_Q^{(i,j)}, \quad K^{(i,j)} = X W_K^{(i,j)}, \quad V^{(i,j)} = X W_V^{(i,j)}$$

Then computes attention:
$$\text{Attention}^{(i,j)} = \text{softmax}\left(\frac{Q^{(i,j)} (K^{(i,j)})^\top}{\sqrt{d_s}}\right) V^{(i,j)}$$

#### Result Aggregation
1. Concatenate dimension slices within head groups: $\text{HeadGroup}_i = \text{Concat}_{j=1}^m \text{Attention}^{(i,j)}$
2. Concatenate all head groups: $\text{Output} = \text{Concat}_{i=1}^n \text{HeadGroup}_i$

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16)
- **Models**: 4-layer Dense Transformer, 4-layer MoE Transformer (8 experts/layer)
- **Parameters**: Batch size 1024, 16 heads, 512 head dimension, 32768 MLP hidden size

### Baseline vs Proposed
- **Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- **Proposed**: Two-level partitioning (m×n=16)

### Results
| Model Type    | Method                | TPS (tokens/sec) | TPOT (ms) |
| ------------- | --------------------- | ---------------- | --------------- |
| 4-layer Dense | Baseline (TP=8, PP=2) | 1,200,000        | 0.35            |
| 4-layer Dense | Proposed (m×n=16)     | 1,580,000        | 0.22            |
| 4-layer MoE   | Baseline (TP=8, PP=2) | 850,000          | 0.45            |
| 4-layer MoE   | Proposed (m×n=16)     | 1,150,000        | 0.30            |

### Performance Improvements
- **Dense model**: +31.7% throughput, -37.1% communication overhead
- **MoE model**: +35.3% throughput, -33.3% communication overhead

## DAG Generation Components

### For Method Implementation
1. **Input Distribution**: Split input $X$ across $m \times n$ devices
2. **Weight Partitioning**: Create $W^{(i,j)}$ blocks for each partition
3. **Parallel Projection**: Compute $Q^{(i,j)}, K^{(i,j)}, V^{(i,j)}$ on each device
4. **Local Attention**: Compute $\text{Attention}^{(i,j)}$ on each device
5. **Hierarchical Concatenation**: 
   - Concatenate dimension slices within head groups
   - Concatenate head groups for final output

### For Experimental Deployment
1. **Environment Setup**: Configure 16 H100 GPUs with FP16 precision
2. **Model Loading**: Load 4-layer Dense or MoE Transformer
3. **Partitioning Configuration**: Set m×n=16 partitions
4. **Device Mapping**: Assign partitions to GPUs
5. **Performance Monitoring**: Measure TPS and TPOT metrics
6. **Verification**: Ensure proper load balancing and communication patterns

## Conclusion
Our two-level partitioning method enables efficient deployment of MHA computations across $m \times n$ devices, achieving up to 35% throughput improvement and 37% communication overhead reduction. This approach provides a scalable solution for very large transformer models by combining head-wise and intra-head dimension-wise slicing.