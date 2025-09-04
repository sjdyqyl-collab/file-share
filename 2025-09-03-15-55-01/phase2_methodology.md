# Helix: Two-Level Attention Partitioning - Methodology

## Multi-Head Attention Recap
Given input tensor $X \in \mathbb{R}^{B \times L \times D}$ where:
- B: batch size
- L: sequence length  
- D: embedding dimension

MHA projects X into query, key, value tensors:
$$Q, K, V = XW_Q, XW_K, XW_V$$
where $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$

Embedding dimension D splits into h heads, each with dimension $d = D/h$

Each head i performs scaled dot-product attention:
$$\text{Attention}_i(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d}}\right) V_i$$

## Two-Level Partitioning Scheme

### Parameters
- h: total number of heads (16)
- d: dimension per head (512)
- n: number of head partitions
- m: number of dimension partitions per head
- $h_g = h/n$: heads per group
- $d_s = d/m$: slice dimension per partition
- Total partitions: m × n

### Step 1: Partition Weight Matrices
Each projection matrix $W \in \mathbb{R}^{D \times D}$ (for Q, K, V) is partitioned into blocks $W^{(i,j)}$ where:
- $i \in [1, n]$: head group index
- $j \in [1, m]$: intra-head dimension slice index
- $W^{(i,j)} \in \mathbb{R}^{d_s \cdot h_g \times d_s \cdot h_g}$

### Step 2: Computation on Each Partition
Each device handling partition $(i,j)$ computes:
$$Q^{(i,j)} = X W_Q^{(i,j)}$$
$$K^{(i,j)} = X W_K^{(i,j)}$$
$$V^{(i,j)} = X W_V^{(i,j)}$$

Then computes attention:
$$\text{Attention}^{(i,j)} = \text{softmax}\left(\frac{Q^{(i,j)} (K^{(i,j)})^\top}{\sqrt{d_s}}\right) V^{(i,j)}$$

### Step 3: Aggregation of Results
1. Concatenate dimension slices within each head group:
   $$\text{Head}_i = \text{Concat}_{j=1}^m \text{Attention}^{(i,j)}$$
2. Concatenate all head groups:
   $$\text{Output} = \text{Concat}_{i=1}^n \text{Head}_i$$

## Communication Pattern
- **Input distribution**: Each device receives corresponding input slice for projections
- **Intra-group communication**: Devices within same head group communicate for dimension concatenation
- **Inter-group communication**: Final concatenation across head groups if needed
- **Hierarchical partitioning** reduces communication overhead vs naive full-dimension splits

## Implementation Considerations
- Integration with existing model parallel frameworks
- Custom tensor partitioning and communication primitives
- Supports both training and inference (gradient synchronization for training)
- Choice of m and n depends on hardware topology and network bandwidth
- Mixed precision (FP16) recommended for efficiency