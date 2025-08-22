# Phase 2: Methodology Extraction

## Two-Level Partitioning Method

### Overview
The proposed method partitions MHA along two dimensions:
1. **Head Dimension Partitioning**: Divide h heads into n groups (h_g = h/n heads per group)
2. **Intra-Head Dimension Partitioning**: Split each head's feature dimension d into m segments (d_s = d/m per segment)

This creates m×n total partitions that can be assigned to m×n devices.

### Mathematical Formulation

#### Input Representation
- Input tensor: X ∈ ℝ^(B×L×D)
- B: batch size, L: sequence length, D: embedding dimension
- h: number of heads, d: dimension per head (D = h×d)

#### Weight Matrix Partitioning
Each projection matrix W ∈ ℝ^(D×D) (for Q, K, V) is partitioned into blocks W^(i,j) where:
- i ∈ [1,n]: indexes head group
- j ∈ [1,m]: indexes intra-head dimension slice
- Each block: W^(i,j) ∈ ℝ^(d_s×h_g × d_s×h_g)

#### Per-Partition Computation
Each device handling partition (i,j) computes:
```
Q^(i,j) = X W_Q^(i,j)
K^(i,j) = X W_K^(i,j)  
V^(i,j) = X W_V^(i,j)

Attention^(i,j) = softmax(Q^(i,j) (K^(i,j))^T / √d_s) V^(i,j)
```

#### Aggregation Process
1. **Dimension-wise concatenation**: Concatenate m dimension slices within each head group
2. **Head-wise concatenation**: Concatenate n head groups to reconstruct full MHA output

```
Output = Concat_{i=1}^n (Concat_{j=1}^m Attention^(i,j))
```

### Communication Pattern
- **Input distribution**: Each device receives corresponding input slice for projections
- **Intra-group communication**: Devices within same head group communicate for dimension concatenation
- **Final assembly**: Head group outputs concatenated without additional communication if properly placed
- **Hierarchical structure**: Reduces communication overhead compared to naive full-dimension splits

### Implementation Details
- Integrates with existing model parallel frameworks
- Supports both training and inference by adapting gradient synchronization
- Choice of m and n depends on hardware topology and network bandwidth
- Compatible with mixed precision (FP16) computation
- Works with standard transformer architectures (Dense and MoE)

### Advantages
- **Scalability**: Supports m×n devices, exceeding head-wise splitting limits
- **Load Balancing**: Even division by balancing head count and feature dimension
- **Memory Efficiency**: Each device stores fraction of parameters and activations
- **Communication Efficiency**: Localized partitions reduce cross-device synchronization