# Phase 2: Methodology Extraction

## Two-Level Partitioning Method Overview

The proposed method partitions the Multi-Head Attention (MHA) mechanism along two dimensions:

### 1. Head Dimension Partitioning
- Total h heads divided into n groups
- Each group contains h_g = h/n heads

### 2. Intra-Head Dimension Partitioning  
- Each head's feature dimension d sliced into m segments
- Each segment has d_s = d/m dimensions

This creates m×n total partitions, each corresponding to a (head group, dimension slice) pair.

## Mathematical Formulation

### Input and Projections
- Input tensor: X ∈ ℝ^(B×L×D) where B=batch size, L=sequence length, D=embedding dimension
- Weight matrices: W_Q, W_K, W_V ∈ ℝ^(D×D)
- Total dimension: D = h×d (h heads × d dimensions per head)

### Partitioning Weight Matrices
Each projection matrix W is partitioned into blocks W^(i,j) where:
- i ∈ [1,n] indexes head group
- j ∈ [1,m] indexes intra-head dimension slice
- W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g)

### Computation Per Partition
Each device handling partition (i,j) computes:
```
Q^(i,j) = X W_Q^(i,j)
K^(i,j) = X W_K^(i,j)  
V^(i,j) = X W_V^(i,j)

Attention^(i,j) = softmax(Q^(i,j) (K^(i,j))^T / √d_s) V^(i,j)
```

### Result Aggregation
1. Concatenate dimension slices j=1,...,m within each head group i along feature dimension
2. Concatenate outputs from all head groups i=1,...,n along head dimension

Final output: `Output = Concat_i=1^n (Concat_j=1^m Attention^(i,j))`

## Communication Pattern
- Input slices distributed to corresponding devices
- Partial results concatenated within head groups
- Final head group outputs concatenated with minimal communication
- Hierarchical partitioning reduces communication overhead

## Implementation Parameters
- h = 16 (number of heads in experiments)
- d = 512 (dimension per head in experiments)
- D = 8192 (total embedding dimension = h×d)
- m×n = 16 (total partitions matching 16 GPUs)
- Batch size B = 1024
- Sequence length L varies by task
- Precision: FP16 mixed precision

## Advantages
- **Scalability**: Supports m×n devices beyond head count h
- **Load Balancing**: Even division of heads and dimensions
- **Memory Efficiency**: Each device stores fraction of parameters
- **Communication Efficiency**: Localized partitions reduce bandwidth