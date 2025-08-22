# Helix: Two-Level Attention Partitioning for Distributed Transformer Models

### Abstract
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Introduction
Transformer architectures with multi-head attention (MHA) are growing exponentially, requiring efficient distributed deployment. Traditional MHA parallelization splits attention heads across devices, but this approach is limited when the number of devices exceeds the number of heads, leading to suboptimal utilization and communication bottlenecks.

## Method

### Two-Level Partitioning Scheme
Our method partitions MHA along two dimensions:
1. **Head Dimension Partitioning**: Split h heads into n groups (h/n heads per group)
2. **Intra-Head Dimension Partitioning**: Split each head's feature dimension d into m segments (d/m dimensions per segment)

This creates m×n total partitions, each handling a distinct (head group, dimension slice) pair.

### Technical Implementation
Given input tensor $X \in \mathbb{R}^{B \times L \times D}$:

**Parameters:**
- h: total heads
- d: dimension per head (D = h×d)
- n: head partitions
- m: dimension partitions per head
- h_g = h/n: heads per group
- d_s = d/m: slice dimension

**Weight Partitioning:**
Each projection matrix W is partitioned into blocks $W^{(i,j)}$ where:
- i ∈ [1,n]: head group index
- j ∈ [1,m]: dimension slice index
- $W^{(i,j)} \in \mathbb{R}^{d_s \cdot h_g \times d_s \cdot h_g}$

**Computation per partition (i,j):**
$$Q^{(i,j)} = X W_Q^{(i,j)}, \quad K^{(i,j)} = X W_K^{(i,j)}, \quad V^{(i,j)} = X W_V^{(i,j)}$$
$$\text{Attention}^{(i,j)} = \text{softmax}\left(\frac{Q^{(i,j)} (K^{(i,j)})^\top}{\sqrt{d_s}}\right) V^{(i,j)}$$

**Aggregation:**
1. Concatenate dimension slices within each head group
2. Concatenate head groups to form final MHA output

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16)
- **Models**: 4-layer Dense Transformer, 4-layer MoE Transformer (8 experts/layer)
- **Fixed params**: Batch=1024, Heads=16, Head-dim=512, MLP-hidden=32768

### Baseline vs Proposed
| Model | Method | TPS (tokens/sec) | TPOT (ms) |
|-------|--------|------------------|-----------|
| Dense | Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| Dense | Proposed (m×n=16) | 1,580,000 | 0.22 |
| MoE   | Baseline (TP=8, PP=2) | 850,000 | 0.45 |
| MoE   | Proposed (m×n=16) | 1,150,000 | 0.30 |

### Results
- **Dense model**: 31.7% throughput improvement, 37.1% overhead reduction
- **MoE model**: 35.3% throughput improvement, 33.3% overhead reduction
- Full hardware utilization achieved with m×n=16 configuration

## Conclusion
The two-level partitioning method enables efficient deployment of MHA computations across m×n devices, significantly improving scalability beyond traditional head-wise splitting. Experimental results demonstrate substantial throughput improvements (up to 35%) and communication overhead reduction (over 30%) on 16 GPUs.