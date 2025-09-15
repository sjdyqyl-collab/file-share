# Helix: Two-Level Attention Partitioning for Large-Scale Transformer Models

### Abstract

We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## 1. Introduction

Transformer architectures with multi-head attention (MHA) face scaling challenges as model sizes grow exponentially. Traditional MHA parallelization splits attention heads across devices, but this becomes limiting when the number of devices exceeds available heads, leading to suboptimal utilization and communication bottlenecks.

We introduce a two-level partitioning strategy that extends beyond conventional head-wise splitting by further segmenting each attention head's internal dimension. Specifically, we partition the MHA layer into *n* head groups and *m* dimension slices per head, resulting in *m × n* partitions mapped onto *m × n* devices.

## 2. Method

### 2.1 Multi-Head Attention Background

Given input tensor X ∈ ℝ^(B×L×D) where B=batch size, L=sequence length, D=embedding dimension:
- MHA projects X into Q, K, V using weight matrices W_Q, W_K, W_V ∈ ℝ^(D×D)
- D is split into h heads, each with dimension d = D/h
- Each head i computes: Attention_i = softmax(Q_i K_i^T/√d) V_i

### 2.2 Two-Level Partitioning Scheme

**Parameters:**
- h = number of heads
- d = dimension per head  
- n = head partitions
- m = dimension partitions per head
- h_g = h/n = heads per group
- d_s = d/m = slice dimension per partition

**Partitioning Process:**
1. **Head-level**: Divide h heads into n groups (h_g heads each)
2. **Dimension-level**: Slice each head's d dimensions into m segments (d_s each)
3. **Result**: m×n total partitions, each handling h_g heads × d_s dimensions

**Weight Matrix Partitioning:**
Each projection matrix W ∈ ℝ^(D×D) is partitioned into blocks W^(i,j) where:
- i ∈ [1,n] indexes head group
- j ∈ [1,m] indexes dimension slice
- W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g)

**Computation Per Partition:**
Each device (i,j) computes:
```
Q^(i,j) = X W_Q^(i,j)
K^(i,j) = X W_K^(i,j)
V^(i,j) = X W_V^(i,j)
Attention^(i,j) = softmax(Q^(i,j)(K^(i,j))^T/√d_s) V^(i,j)
```

**Result Aggregation:**
1. Concatenate dimension slices within each head group
2. Concatenate outputs from all head groups
3. Final output matches original MHA dimension

### 2.3 Advantages
- **Scalability**: Supports m×n devices beyond head count h
- **Load Balancing**: Even division of heads and dimensions
- **Memory Efficiency**: Each device stores fraction of parameters
- **Communication Efficiency**: Localized partitions reduce bandwidth

## 3. Experiments

### 3.1 Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Model**: 2-layer Dense Transformer
- **Configuration**: h=16 heads, d=512 per head, D=8192 total, MLP hidden=32768
- **Batch Size**: 1024
- **Precision**: FP16

### 3.2 Baseline vs Proposed

| Method | TPS (tokens/sec) | TPOT (ms) |
|--------|------------------|-----------|
| Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| Proposed (m×n=16) | 1,580,000 | 0.22 |

**Improvements:**
- Throughput: +31.7% (1.2M → 1.58M tokens/sec)
- Overhead: -37.1% (0.35 → 0.22 ms)

### 3.3 Analysis
The proposed method fully utilizes all 16 GPUs through m×n=16 partitions, achieving better hardware utilization than the baseline TP=8+PP=2 approach. The finer granularity enables improved load balancing and reduced cross-device communication.

## 4. Conclusion

We presented a novel two-level partitioning method for MHA that combines head-wise and intra-head dimension-wise slicing. Experiments on 16 GPUs demonstrate substantial improvements in inference throughput (31.7%) while reducing communication overhead (37.1%). This approach provides a promising pathway for efficient distributed deployment of massive transformer models, scaling beyond traditional head-count limitations.

## Key Implementation Parameters
- Total partitions: m×n = 16 (4 head groups × 4 dimension slices)
- Heads per group: h_g = 4
- Dimensions per slice: d_s = 128
- Partition size: 4 heads × 128 dimensions = 512 dimensions per device