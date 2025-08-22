# Helix: Two-Level Attention Partitioning - Methodology

## Abstract (Retained in full)
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Methodology

### Multi-Head Attention Recap
Given input tensor $X \in \mathbb{R}^{B \times L \times D}$ where:
- $B$ = batch size
- $L$ = sequence length  
- $D$ = embedding dimension

MHA layer projects $X$ into query, key, value tensors:
$$Q, K, V = XW_Q, XW_K, XW_V$$
where $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$

Embedding dimension $D$ split into $h$ heads, each with dimension $d = D/h$

Each head $i$ performs:
$$\text{Attention}_i(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d}}\right) V_i$$

### Two-Level Partitioning Scheme

#### Parameters:
- $h$: total number of heads
- $d$: dimension per head ($D = h \times d$)
- $n$: number of head partitions
- $m$: number of dimension partitions per head
- $h_g = h/n$: heads per group
- $d_s = d/m$: slice dimension per partition

#### Step 1: Partition Weight Matrices
Each projection matrix $W \in \mathbb{R}^{D \times D}$ (for Q, K, V) partitioned into blocks $W^{(i,j)}$ where:
- $i \in [1, n]$: head group index
- $j \in [1, m]$: intra-head dimension slice index
- $W^{(i,j)} \in \mathbb{R}^{d_s \cdot h_g \times d_s \cdot h_g}$

#### Step 2: Distributed Computation
Each device handling partition $(i,j)$ computes:
$$Q^{(i,j)} = X W_Q^{(i,j)}, \quad K^{(i,j)} = X W_K^{(i,j)}, \quad V^{(i,j)} = X W_V^{(i,j)}$$

Then computes attention for assigned slice:
$$\text{Attention}^{(i,j)} = \text{softmax}\left(\frac{Q^{(i,j)} (K^{(i,j)})^\top}{\sqrt{d_s}}\right) V^{(i,j)}$$

#### Step 3: Result Aggregation
1. Concatenate dimension slices within each head group:
   $$\text{HeadGroup}_i = \text{Concat}_{j=1}^m \text{Attention}^{(i,j)}$$
2. Concatenate all head groups:
   $$\text{Output} = \text{Concat}_{i=1}^n \text{HeadGroup}_i$$

### Communication Pattern
- **Input distribution**: Each device receives corresponding input slice for projections
- **Intra-group communication**: Devices within same head group communicate to concatenate dimension slices
- **Final aggregation**: Head group outputs concatenated without additional communication if properly placed

### Implementation Details
- Compatible with existing model parallel frameworks
- Supports both training and inference
- Choice of $m$ and $n$ depends on hardware topology and network bandwidth
- Uses hierarchical partitioning to reduce communication overhead vs naive full-dimension splits

### DAG Components for Deployment
1. **Input Splitting**: Distribute $X$ to all $m \times n$ devices
2. **Projection**: Parallel matrix multiplication $XW^{(i,j)}$ on each device
3. **Attention Computation**: Local attention computation on each partition
4. **Intra-group Concatenation**: Concatenate dimension slices within head groups
5. **Inter-group Concatenation**: Concatenate head groups for final output
6. **Output Collection**: Gather final result from all devices