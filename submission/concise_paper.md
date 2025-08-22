# Helix: Two-Level Attention Partitioning for Large-Scale Transformer Deployment

### Abstract

We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## 1. Introduction

Transformer architectures with multi-head attention (MHA) require efficient distribution across hardware as model sizes grow exponentially. Traditional MHA parallelization splits attention heads across devices, but this becomes suboptimal when device count exceeds head count, causing communication bottlenecks.

We introduce a two-level partitioning strategy that extends beyond head-wise splitting by segmenting each attention head's internal dimension. This creates *m × n* partitions (head groups × dimension slices) that can be mapped to *m × n* devices, enabling flexible scaling and better memory distribution.

## 2. Method

### 2.1 Two-Level Partitioning Scheme

Our method partitions MHA along two dimensions:
1. **Head Dimension Partitioning**: Divide *h* heads into *n* groups (h_g = h/n heads per group)
2. **Intra-Head Dimension Partitioning**: Split each head's feature dimension *d* into *m* segments (d_s = d/m per segment)

This creates *m × n* total partitions assignable to *m × n* devices.

### 2.2 Mathematical Formulation

**Input**: X ∈ ℝ^(B×L×D) where B=batch size, L=sequence length, D=embedding dimension
**Parameters**: h=heads, d=dimension per head, D=h×d

**Weight Matrix Partitioning**:
Each projection matrix W ∈ ℝ^(D×D) (for Q,K,V) is partitioned into blocks W^(i,j):
- i ∈ [1,n]: head group index
- j ∈ [1,m]: dimension slice index
- Each block: W^(i,j) ∈ ℝ^(d_s×h_g × d_s×h_g)

**Per-Partition Computation**:
```
Q^(i,j) = X W_Q^(i,j)
K^(i,j) = X W_K^(i,j)
V^(i,j) = X W_V^((i,j))

Attention^(i,j) = softmax(Q^(i,j) (K^(i,j))^T / √d_s) V^(i,j)
```

**Aggregation**:
1. Concatenate m dimension slices within each head group
2. Concatenate n head groups to reconstruct full MHA output

### 2.3 Communication Pattern
- Hierarchical communication reduces overhead
- Intra-group communication for dimension concatenation
- No additional communication for head group concatenation if properly placed
- Each device stores fraction of parameters and activations

## 3. Experiments

### 3.1 Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16)
- **Batch Size**: 1024
- **Models**: 4-layer Dense Transformer, 4-layer MoE Transformer (8 experts/layer)
- **Parameters**: 16 heads, 512 dim/head, 32768 MLP hidden size
- **Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)

### 3.2 Results

| Model | Method | TPS (tokens/sec) | TPOT (ms) | Improvement |
|-------|--------|------------------|-----------|-------------|
| Dense | Baseline | 1,200,000 | 0.35 | - |
| Dense | Proposed | 1,580,000 | 0.22 | +31.7% TPS, -37.1% TPOT |
| MoE | Baseline | 850,000 | 0.45 | - |
| MoE | Proposed | 1,150,000 | 0.30 | +35.3% TPS, -33.3% TPOT |

### 3.3 Analysis
- Consistent improvements across both dense and MoE architectures
- Higher gains for MoE models (35.3% vs 31.7%)
- Better hardware utilization through m×n=16 partitions
- Reduced synchronization costs and efficient communication patterns

## 4. Conclusion

The proposed two-level partitioning method enables efficient deployment of MHA computations across *m × n* devices by combining head-wise and intra-head dimension-wise slicing. Experiments demonstrate substantial improvements in inference throughput (up to 35%) while reducing communication overhead by over 30%. This approach offers a promising direction for efficient distributed inference of large transformer architectures.

## Deployment DAG Information

For generating the directed acyclic graph for experimental model deployment:

**Partition Configuration**:
- Total devices: m×n (e.g., 16 for experiments)
- Head groups: n
- Dimension slices: m
- Partition mapping: Each (head_group, dimension_slice) → device

**Communication Flow**:
1. Input distribution to all devices
2. Local computation per partition
3. Intra-group concatenation (devices in same head group)
4. Final output assembly (no communication if properly placed)

**Implementation Requirements**:
- Custom tensor partitioning primitives
- Hierarchical communication support
- Gradient synchronization for training
- Hardware topology awareness for optimal m,n selection