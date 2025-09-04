# Helix: Two-Level Attention Partitioning for Large-Scale Transformer Models

### Abstract
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Method

### Two-Level Partitioning Scheme
Our method partitions MHA along two dimensions:
1. **Head Dimension Partitioning**: h heads divided into n groups (h_g = h/n heads per group)
2. **Intra-Head Dimension Partitioning**: Each head's dimension d sliced into m segments (d_s = d/m per segment)

### Mathematical Formulation
Given input tensor $X \in \mathbb{R}^{B \times L \times D}$:
- Partition weight matrices $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$ into blocks $W^{(i,j)}$
- Each device computes: $Q^{(i,j)} = X W_Q^{(i,j)}$, $K^{(i,j)} = X W_K^{(i,j)}$, $V^{(i,j)} = X W_V^{(i,j)}$
- Attention computation: $\text{Attention}^{(i,j)} = \text{softmax}\left(\frac{Q^{(i,j)} (K^{(i,j)})^\top}{\sqrt{d_s}}\right) V^{(i,j)}$
- Aggregation: $\text{Output} = \text{Concat}_{i=1}^n \left( \text{Concat}_{j=1}^m \text{Attention}^{(i,j)} \right)$

### Key Parameters
- Total partitions: m × n
- Heads per group: h_g = h/n
- Dimension per slice: d_s = d/m
- Communication: Hierarchical with intra-group and inter-group concatenation

## Implementation Notes
- Integration with existing model parallel frameworks through customized tensor partitioning and communication primitives
- Supports both training and inference by adapting gradient synchronization
- Choice of m and n depends on hardware topology and network bandwidth considerations
- Mixed precision (FP16) recommended for efficiency

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Model**: 4-layer Dense Transformer
- **Config**: 16 heads, 512 head dimension, 32768 MLP hidden size, batch size 1024, FP16

### Results
| Method | TPS (tokens/sec) | TPOT (ms) |
|--------|------------------|-----------|
| Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| Proposed (m×n=16) | 1,580,000 | 0.22 |

### Performance Gains
- **Throughput**: +31.7% (1.2M → 1.58M tokens/sec)
- **Overhead**: -37.1% (0.35ms → 0.22ms TPOT)

## Conclusion
The two-level partitioning method enables efficient deployment of MHA across m×n devices, achieving significant improvements in throughput and communication efficiency compared to traditional parallelization approaches.