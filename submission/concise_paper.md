# Helix: Two-Level Attention Partitioning for Large-Scale Transformers

### Abstract
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## 1. Introduction
Transformer architectures with multi-head attention (MHA) have become state-of-the-art, but their exponential growth requires efficient distributed computation. Traditional MHA parallelization splits attention heads across devices, leading to suboptimal utilization when devices > heads. We introduce a two-level partitioning strategy that segments both heads and internal dimensions, enabling flexible scaling and better memory distribution across *m × n* devices.

## 2. Method

### 2.1 Two-Level Partitioning Overview
Our method partitions MHA along two dimensions:
- **Head Dimension**: h heads → n groups (h_g = h/n heads per group)
- **Intra-Head Dimension**: d dimensions → m segments (d_s = d/m per segment)

Result: *m × n* partitions mapped to *m × n* devices.

### 2.2 Technical Details
**Input**: X ∈ ℝ^(B×L×D) where B=batch, L=sequence, D=embedding
**Parameters**: h heads, d=D/h dimensions per head
**Partitioning**: n head groups, m dimension slices per head

**Weight matrices** W_Q, W_K, W_V ∈ ℝ^(D×D) are partitioned into blocks W^(i,j) where:
- i ∈ [1,n] indexes head group
- j ∈ [1,m] indexes dimension slice
- Each block: ℝ^(d_s·h_g × d_s·h_g)

**Per-partition computation**:
```
Q^(i,j) = X W_Q^(i,j)
K^(i,j) = X W_K^(i,j)  
V^(i,j) = X W_V^(i,j)
Attention^(i,j) = softmax(Q^(i,j)(K^(i,j))^T/√d_s) V^(i,j)
```

**Aggregation**:
1. Concatenate dimension slices within each head group
2. Concatenate head group outputs along head dimension

### 2.3 Advantages
- **Scalability**: Supports m×n devices beyond head count limits
- **Load Balancing**: Even division of heads and dimensions
- **Memory Efficiency**: Each device stores fraction of parameters
- **Communication Efficiency**: Localized partitions reduce synchronization

## 3. Experiments

### 3.1 Setup
- **Hardware**: 16 NVIDIA H100 GPUs, FP16
- **Models**: 2-layer Dense Transformer, 2-layer MoE (4 experts/layer)
- **Fixed**: 16 heads, 512 head dimension, 32768 MLP hidden, batch=1024

### 3.2 Results
| Model | Method | TPS (tokens/sec) | TPOT (ms) |
|-------|--------|------------------|-----------|
| Dense | Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| Dense | Proposed (m×n=16) | 1,580,000 | 0.22 |
| MoE   | Baseline (TP=8, PP=2) | 850,000 | 0.45 |
| MoE   | Proposed (m×n=16) | 1,150,000 | 0.30 |

### 3.3 Analysis
- **Dense model**: 31.7% throughput increase, 37.1% overhead reduction
- **MoE model**: 35.3% throughput increase, 33.3% overhead reduction
- Fine-grained partitioning enables better load balancing and reduced communication vs TP+PP baseline

## 4. Conclusion
We propose a two-level partitioning method that combines head-wise and intra-head dimension-wise slicing for MHA layers. Experiments on 16 GPUs demonstrate up to 35% throughput improvement and 30%+ communication overhead reduction. This approach offers a promising direction for efficient distributed inference of large transformer architectures.

## References
[1] Original paper: Helix: Two-Level Attention Partitioning for Large-Scale Transformers