# Helix: Two-Level Attention Partitioning for Large-Scale Transformers

### Abstract
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Introduction

Transformer architectures employing multi-head attention (MHA) have become fundamental to state-of-the-art models. As model sizes grow exponentially, efficient distribution across hardware units becomes critical. Traditional MHA parallelization splits attention heads across devices, but this approach is limited by the fixed number of heads and leads to suboptimal utilization when devices exceed heads.

We introduce a novel two-level partitioning strategy that extends beyond conventional head-wise splitting by further segmenting each attention head's internal dimension. The method partitions MHA layers into *n* head groups and *m* dimension slices per head, resulting in *m × n* partitions mapped onto *m × n* devices. This enables flexible scaling, better memory distribution, and reduced inter-device communication.

## Method

### Two-Level Partitioning Overview
Our method partitions MHA along two dimensions:
1. **Head Dimension Partitioning**: h heads divided into n groups, each containing h/n heads
2. **Intra-Head Dimension Partitioning**: Each head's feature dimension d sliced into m segments of size d/m

This creates m × n partitions, each handling a (head group, dimension slice) pair.

### Mathematical Formulation
Given input X ∈ ℝ^(B×L×D) where B=batch, L=sequence, D=embedding:
- h = number of heads
- d = dimension per head (D = h×d)
- n = head partitions
- m = dimension partitions per head
- h_g = h/n (heads per group)
- d_s = d/m (slice dimension per partition)

Each projection matrix W ∈ ℝ^(D×D) is partitioned into blocks W^(i,j) where i∈[1,n] indexes head group and j∈[1,m] indexes dimension slice:
```
W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g)
```

### Computation per Partition
Each device handling partition (i,j) computes:
```
Q^(i,j) = X W_Q^(i,j)
K^(i,j) = X W_K^(i,j)
V^(i,j) = X W_V^(i,j)
Attention^(i,j) = softmax((Q^(i,j)(K^(i,j))^T)/√d_s) V^(i,j)
```

### Aggregation
1. Concatenate dimension slices within each head group
2. Concatenate outputs from all head groups to reconstruct full MHA output

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Precision**: FP16
- **Models**: 2-layer Dense Transformer, 2-layer MoE Transformer (4 experts/layer)
- **Fixed Parameters**: Batch=1024, Heads=16, Dim/head=512, MLP hidden=32768

### Baseline
Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2) on 16 GPUs

### Results

| Model | Method | TPS (tokens/sec) | TPOT (ms) |
|-------|--------|------------------|-----------|
| Dense | Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| Dense | Proposed (m×n=16) | 1,580,000 | 0.22 |
| MoE | Baseline (TP=8, PP=2) | 850,000 | 0.45 |
| MoE | Proposed (m×n=16) | 1,150,000 | 0.30 |

### Analysis
- **Dense model**: 31.7% throughput improvement, 37.1% overhead reduction
- **MoE model**: 35.3% throughput improvement, 33.3% overhead reduction
- Full utilization of 16 GPUs through m×n=16 partitions
- Reduced synchronization cost and efficient communication patterns

## Conclusion

Our two-level partitioning method enables deployment of MHA computations across m×n devices by combining head-wise and intra-head dimension-wise slicing. Experiments demonstrate substantial improvements in inference throughput (up to 35%) and communication overhead reduction (over 30%) compared to tensor+pipeline parallelism baselines. This approach provides a promising pathway for efficient distributed deployment of massive transformer architectures.

## Deployment Configuration

### Proposed Method Parameters
- **Partitioning**: m=4, n=4 (m×n=16 total partitions)
- **Head groups**: 4 groups × 4 heads/group = 16 heads
- **Dimension slices**: 4 slices × 128 dimensions/slice = 512 dimensions/head
- **Device mapping**: Each partition (i,j) → device_id = i×4 + j

### Baseline Method Parameters
- **Tensor Parallelism**: TP=8 across devices [0-7] and [8-15]
- **Pipeline Parallelism**: PP=2 with stage 0 (layers 0) and stage 1 (layers 1)
- **Total devices**: 16 (8×2 configuration)