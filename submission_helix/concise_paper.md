# Helix: Two-Level Attention Partitioning for Large Transformer Models

### Abstract

We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Introduction

Transformer architectures with multi-head attention (MHA) have become state-of-the-art, but traditional MHA parallelization only splits attention heads across devices. This approach leads to suboptimal utilization when the number of devices exceeds the number of heads. We introduce a two-level partitioning strategy that extends beyond conventional head-wise splitting by further segmenting each attention head's internal dimension, enabling flexible scaling and better memory distribution.

## Method

### Two-Level Partitioning Scheme

Given an input tensor X ∈ ℝ^(B×L×D) where B is batch size, L is sequence length, and D is embedding dimension:

**Parameters:**
- h: number of attention heads (16)
- d: dimension per head (512)
- D: total embedding dimension = h × d = 8192
- n: number of head partitions (groups)
- m: number of dimension partitions per head
- h_g = h/n: heads per group
- d_s = d/m: dimension slice per partition

**Partitioning:**
1. **Head Dimension Partitioning**: Divide h heads into n groups, each containing h/n heads
2. **Intra-Head Dimension Partitioning**: Split each head's d dimensions into m segments
3. **Total Partitions**: m × n partitions for m × n devices

**Weight Matrix Partitioning:**
Each projection matrix W_Q, W_K, W_V ∈ ℝ^(D×D) is partitioned into blocks W^(i,j) where:
- i ∈ [1,n]: head group index
- j ∈ [1,m]: dimension slice index
- Each block: W^(i,j) ∈ ℝ^(d_s×h_g × d_s×h_g)

**Computation per Partition:**
Each device (i,j) computes:
- Q^(i,j) = X W_Q^(i,j), K^(i,j) = X W_K^(i,j), V^(i,j) = X W_V^(i,j)
- Attention^(i,j) = softmax(Q^(i,j) (K^(i,j))^T / √d_s) V^(i,j)

**Aggregation:**
1. Concatenate dimension slices within each head group: Concat_j=1^m Attention^(i,j)
2. Concatenate head groups: Concat_i=1^n (Concat_j=1^m Attention^(i,j))

### Implementation Details
- Each device stores 1/(m×n) of total MHA parameters
- Memory footprint per device: O(B×L×d_s×h_g)
- Communication: Intra-group concatenation for dimension slices, final concatenation for head groups
- Compatible with existing model parallel frameworks

## Experiments

### Setup
- **Hardware**: 16 × NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16)
- **Models**: 
  - 2-layer Dense Transformer
  - 2-layer MoE Transformer (4 experts/layer)
- **Configuration**: 16 heads, 512 head dimension, 8192 hidden size, 32768 MLP hidden size
- **Batch size**: 1024

### Baseline
- Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- 8 GPUs per tensor parallel group, 2 pipeline stages

### Results
| Model | Method | TPS (tokens/sec) | TPOT (ms) |
|-------|--------|------------------|-----------|
| Dense | Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| Dense | Proposed (m×n=16) | 1,580,000 | 0.22 |
| MoE | Baseline (TP=8, PP=2) | 850,000 | 0.45 |
| MoE | Proposed (m×n=16) | 1,150,000 | 0.30 |

### Performance Gains
- **Dense model**: 31.7% throughput increase, 37.1% overhead reduction
- **MoE model**: 35.3% throughput increase, 33.3% overhead reduction

## Conclusion

The two-level partitioning method enables efficient deployment of MHA computations across m×n devices by combining head-wise and intra-head dimension-wise slicing. This approach achieves substantial improvements in inference throughput (up to 35%) while reducing communication overhead by over 30% compared to traditional tensor and pipeline parallelism.