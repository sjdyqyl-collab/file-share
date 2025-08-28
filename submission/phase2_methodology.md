# Phase 2: Methodology Extraction - Helix: Two-Level Attention Partitioning

## Method Overview
Proposed **two-level partitioning method** for Multi-Head Attention (MHA) mechanism that extends beyond conventional head-wise splitting by further segmenting each attention head's internal dimension.

## Multi-Head Attention Recap
Given input tensor $X \in \mathbb{R}^{B \times L \times D}$ where:
- B = batch size
- L = sequence length  
- D = embedding dimension

MHA layer projects X into query, key, value tensors:
$$Q, K, V = XW_Q, XW_K, XW_V$$

Where each weight $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$. The embedding dimension D is split into h heads, each with dimension $d = D/h$.

Each head i performs scaled dot-product attention:
$$\text{Attention}_i(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d}}\right) V_i$$

## Two-Level Partitioning Scheme

### Partitioning Dimensions
1. **Head Dimension Partitioning**: Divide h heads into n groups, each containing $\frac{h}{n}$ heads
2. **Intra-Head Dimension Partitioning**: Further slice each head's feature dimension d into m segments, each of size $\frac{d}{m}$

**Result**: $m \times n$ partitions, each corresponding to a distinct (head group, dimension slice) pair

### Parameter Definitions
- h: number of heads
- d: dimension per head, so total $D = h \times d$
- n: number of head partitions
- m: number of dimension partitions per head
- $h_g = \frac{h}{n}$: heads per group
- $d_s = \frac{d}{m}$: slice dimension per partition

### Weight Matrix Partitioning
Each projection matrix $W \in \mathbb{R}^{D \times D}$ (for Q, K, V) is partitioned into blocks $W^{(i,j)}$ where:
- $i \in [1, n]$ indexes the head group
- $j \in [1, m]$ indexes the intra-head dimension slice

Each block:
$$W^{(i,j)} \in \mathbb{R}^{d_s \cdot h_g \times d_s \cdot h_g}$$

### Computation Per Partition
Each device handling partition $(i,j)$ receives corresponding input tensor slices and computes:
$$Q^{(i,j)} = X W_Q^{(i,j)}, \quad K^{(i,j)} = X W_K^{(i,j)}, \quad V^{(i,j)} = X W_V^{(i,j)}$$

Then computes scaled dot-product attention using assigned slice:
$$\text{Attention}^{(i,j)} = \text{softmax}\left(\frac{Q^{(i,j)} (K^{(i,j)})^\top}{\sqrt{d_s}}\right) V^{(i,j)}$$

### Result Aggregation
Since each partition computes attention for subset of heads and dimension slice:

1. **First**: Concatenate dimension slices $j = 1,...,m$ within each head group $i$ along feature dimension to reconstruct full head outputs
2. **Second**: Concatenate outputs from all head groups $i = 1,...,n$ along head dimension to reconstruct full MHA output:

$$\text{Output} = \text{Concat}_{i=1}^n \left( \text{Concat}_{j=1}^m \text{Attention}^{(i,j)} \right)$$

## Communication and Synchronization
- Each device receives corresponding input slice for projections
- Partial results within head group must be concatenated (communication among same group devices)
- After dimension-wise concatenation, final head groups' outputs are concatenated
- Hierarchical partitioning reduces communication overhead vs naive full-dimension splits

## Advantages
- **Scalability**: Supports deployment on $m \times n$ devices, exceeding head-wise splitting limits
- **Load Balancing**: Even division of both head count and feature dimension
- **Reduced Memory**: Each device stores fraction of MHA parameters and intermediate activations
- **Communication Efficiency**: Localized intra-head dimension partitions reduce cross-device synchronization bandwidth

## Implementation Notes
- Integrates with existing model parallel frameworks via custom tensor partitioning and communication primitives
- Supports both training and inference by adapting gradient synchronization
- Choice of m and n depends on hardware topology and network bandwidth