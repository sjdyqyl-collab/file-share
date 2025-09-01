# Methodology: Two-Level Attention Partitioning

## Overview
Our proposed **two-level partitioning method** for Multi-Head Attention (MHA) mechanism enables finer-grained distribution of computation by partitioning both attention heads and their internal dimensions. This creates m × n partitions that can be mapped to m × n devices.

## Multi-Head Attention Background
Given input tensor X ∈ ℝ^(B×L×D) where:
- B = batch size
- L = sequence length  
- D = embedding dimension

MHA projects X into query, key, and value tensors:
```
Q, K, V = XW_Q, XW_K, XW_V
```
where W_Q, W_K, W_V ∈ ℝ^(D×D)

The embedding dimension D is split into h heads, each with dimension d = D/h

Each head i performs scaled dot-product attention:
```
Attention_i(Q_i, K_i, V_i) = softmax((Q_i K_i^T)/√d) V_i
```

## Two-Level Partitioning Scheme

### Level 1: Head Dimension Partitioning
- Total h heads divided into n groups
- Each group contains h_g = h/n heads

### Level 2: Intra-Head Dimension Partitioning
- Each head's feature dimension d sliced into m segments
- Each segment has size d_s = d/m

### Result
- Total partitions: m × n
- Each partition corresponds to a (head group, dimension slice) pair
- Each partition handles d_s × h_g dimensions

## Detailed Partitioning

### Weight Matrix Partitioning
Each projection matrix W ∈ ℝ^(D×D) (for Q, K, V) is partitioned into blocks W^(i,j) where:
- i ∈ [1,n] indexes head group
- j ∈ [1,m] indexes intra-head dimension slice

Each block:
```
W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g)
```

### Computation per Partition
Each device handling partition (i,j) computes:
```
Q^(i,j) = X W_Q^(i,j)
K^(i,j) = X W_K^(i,j)  
V^(i,j) = X W_V^(i,j)

Attention^(i,j) = softmax((Q^(i,j) (K^(i,j))^T)/√d_s) V^(i,j)
```

### Aggregation Process
1. **Intra-group concatenation**: Dimension slices j=1,...,m within each head group i are concatenated along feature dimension
2. **Inter-group concatenation**: Outputs from all head groups i=1,...,n are concatenated along head dimension

```
Output = Concat_{i=1}^n ( Concat_{j=1}^m Attention^(i,j) )
```

## Communication and Synchronization

### Required Communications
1. **Input distribution**: Each device receives corresponding input slice for projections
2. **Intra-group communication**: Partial results within head group must be concatenated
3. **Output assembly**: Final concatenation if devices are not optimally placed

### Communication Efficiency
- Hierarchical partitioning reduces communication overhead
- Localized intra-head dimension partitions minimize cross-device synchronization
- Better than naive full-dimension splits

## Implementation Details

### Integration
- Compatible with existing model parallel frameworks
- Requires customizing tensor partitioning and communication primitives
- Supports both training and inference (with adapted gradient synchronization)

### Parameter Selection
- Choice of m and n depends on hardware topology and network bandwidth
- Must satisfy: h mod n = 0 and d mod m = 0
- Optimal values determined by cluster configuration and model characteristics

### Memory Considerations
- Each device stores only 1/(m×n) of total MHA parameters
- Intermediate activations also partitioned across devices
- Significant memory footprint reduction per device

## Mathematical Formulation Summary

### Dimensions
- h: total number of heads
- d: dimension per head
- D: total embedding dimension = h × d
- n: number of head partitions
- m: number of dimension partitions per head
- h_g = h/n: heads per group
- d_s = d/m: slice dimension per partition

### Partition Mapping
Each device (i,j) handles:
- Head indices: [(i-1)·h_g + 1, i·h_g]
- Dimension indices: [(j-1)·d_s + 1, j·d_s]
- Parameter count: (d_s·h_g)² parameters per weight matrix

### Computational Complexity
Each partition computes attention for:
- Batch size: B
- Sequence length: L
- Head group size: h_g
- Dimension slice: d_s
- Complexity: O(B·L²·h_g·d_s) per partition