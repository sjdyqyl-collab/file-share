# Helix: Two-Level Attention Partitioning - Phase 2: Methodology

## Overview
Proposed **two-level partitioning method** for Multi-Head Attention (MHA) that partitions along both head and intra-head dimensions, creating m × n partitions for m × n devices.

## Mathematical Foundation

### Input Representation
- Input tensor: X ∈ ℝ^(B×L×D)
- B: batch size, L: sequence length, D: embedding dimension
- h: number of heads
- d: dimension per head (D = h × d)
- n: number of head partitions
- m: number of dimension partitions per head
- h_g = h/n: heads per group
- d_s = d/m: slice dimension per partition

### Multi-Head Attention Recap
```
Q, K, V = XW_Q, XW_K, XW_V
W_Q, W_K, W_V ∈ ℝ^(D×D)

Each head i: Attention_i(Q_i, K_i, V_i) = softmax(Q_i K_i^T/√d) V_i
```

## Two-Level Partitioning Scheme

### Step 1: Partition Weight Matrices
Each projection matrix W ∈ ℝ^(D×D) (for Q, K, V) is partitioned into blocks W^(i,j) where:
- i ∈ [1,n]: head group index
- j ∈ [1,m]: intra-head dimension slice index
- W^(i,j) ∈ ℝ^(d_s×h_g × d_s×h_g)

### Step 2: Compute on Each Partition
Each device handling partition (i,j) computes:
```
Q^(i,j) = X W_Q^(i,j)
K^(i,j) = X W_K^(i,j)  
V^(i,j) = X W_V^(i,j)

Attention^(i,j) = softmax(Q^(i,j) (K^(i,j))^T/√d_s) V^(i,j)
```

### Step 3: Hierarchical Aggregation
1. **Intra-group concatenation**: Concatenate m dimension slices within each head group i
2. **Inter-group concatenation**: Concatenate n head groups along head dimension

```
Output = Concat_{i=1}^n ( Concat_{j=1}^m Attention^(i,j) )
```

## Communication Pattern
- **Input distribution**: Each device receives corresponding input slice for projections
- **Intra-group communication**: Required for concatenating dimension slices within head groups
- **Inter-group communication**: Minimal if head groups are placed optimally
- **Hierarchical reduction**: Reduces communication overhead vs. naive full-dimension splits

## Implementation Requirements

### Tensor Partitioning
- Custom partitioning of Q, K, V projection matrices
- Block-wise storage of W_Q, W_K, W_V across devices
- Input tensor slicing according to partition indices

### Communication Primitives
- All-gather operations within head groups for dimension slice concatenation
- Concatenation operations across head groups
- Gradient synchronization for training (extends inference pattern)

### Device Mapping
- Each partition (i,j) maps to a unique device
- Total devices required: m × n
- Placement strategy affects communication efficiency

### Memory Management
- Each device stores: 1/(m×n) of total parameters
- Each device computes: 1/(m×n) of total attention computation
- Intermediate activations: 1/(m×n) of standard MHA memory footprint

## Algorithm Flow
1. **Initialize**: Partition weight matrices into m×n blocks
2. **Distribute**: Assign each partition to a device
3. **Compute**: Each device computes local attention for its partition
4. **Synchronize**: Gather results within head groups, then across groups
5. **Output**: Reconstruct full MHA output from partitioned results

## Advantages
- **Scalability**: m×n devices vs. h heads limitation
- **Load balancing**: Even distribution across both dimensions
- **Memory efficiency**: Linear reduction with partition count
- **Communication efficiency**: Hierarchical aggregation reduces bandwidth