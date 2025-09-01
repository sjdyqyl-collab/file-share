# Phase 2: Methodology Extraction - Two-Level Attention Partitioning

## Overview
Our method introduces a **two-level partitioning scheme** for Multi-Head Attention (MHA) that extends beyond conventional head-wise splitting by further segmenting each attention head's internal dimension. This creates m×n partitions where n=head splits and m=dimension splits per head.

## Multi-Head Attention Background
Given input tensor X ∈ ℝ^(B×L×D):
- B = batch size
- L = sequence length  
- D = embedding dimension

MHA projects X into Q, K, V tensors:
```
Q, K, V = XW_Q, XW_K, XW_V
```
where W_Q, W_K, W_V ∈ ℝ^(D×D)

D is split into h heads, each with dimension d = D/h

Each head i performs:
```
Attention_i(Q_i, K_i, V_i) = softmax(Q_i K_i^T/√d) V_i
```

## Two-Level Partitioning Scheme

### Level 1: Head Dimension Partitioning
- Total h heads divided into n groups
- Each group contains h_g = h/n heads

### Level 2: Intra-Head Dimension Partitioning  
- Each head's feature dimension d sliced into m segments
- Each segment has size d_s = d/m

### Result: m×n Total Partitions
Each partition corresponds to a unique (head_group, dimension_slice) pair

## Detailed Partitioning Implementation

### Weight Matrix Partitioning
Each projection matrix W ∈ ℝ^(D×D) (for Q, K, V) partitioned into blocks W^(i,j) where:
- i ∈ [1,n] indexes head group
- j ∈ [1,m] indexes intra-head dimension slice
- Each block: W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g)

### Input Tensor Partitioning
Each device handling partition (i,j) receives corresponding slices:
```
Q^(i,j) = X W_Q^(i,j)
K^(i,j) = X W_K^(i,j)  
V^(i,j) = X W_V^(i,j)
```

### Computation per Partition
Each device computes:
```
Attention^(i,j) = softmax(Q^(i,j) (K^(i,j))^T/√d_s) V^(i,j)
```

## Result Aggregation
Two-stage concatenation process:

1. **Dimension concatenation**: Concatenate m dimension slices j=1..m within each head group i
2. **Head concatenation**: Concatenate n head groups i=1..n along head dimension

Final output reconstruction:
```
Output = Concat_{i=1}^n (Concat_{j=1}^m Attention^(i,j))
```

## Communication Patterns

### Required Communications:
1. **Input distribution**: Each device receives corresponding input slice for projections
2. **Intra-group synchronization**: Partial results within head group must be concatenated
3. **Minimal inter-group communication**: Final concatenation without additional communication if devices placed accordingly

### Communication Efficiency:
- Hierarchical partitioning reduces communication overhead vs naive full-dimension splits
- Localized intra-head dimension partitions minimize cross-device synchronization bandwidth

## Implementation Details

### Integration Requirements:
- Compatible with existing model parallel frameworks
- Customizable tensor partitioning and communication primitives
- Supports both training and inference (adapts gradient synchronization)

### Parameter Selection:
- Choice of m and n depends on:
  - Hardware topology
  - Network bandwidth
  - Model characteristics
  - Available device count

### Memory Benefits:
- Each device stores only 1/(m×n) of total MHA parameters
- Intermediate activations similarly reduced by factor of m×n
- Enables deployment of larger models within memory constraints