# Phase 2: Methodology Extraction - Helix: Two-Level Attention Partitioning for Distributed Transformer Models

## Abstract (Retained as-is)
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Method Overview

### Multi-Head Attention Recap
Given input tensor $X \in \mathbb{R}^{B \times L \times D}$ where:
- $B$ = batch size
- $L$ = sequence length  
- $D$ = embedding dimension

The MHA layer projects $X$ into query, key, and value tensors:
$$Q, K, V = XW_Q, XW_K, XW_V$$
where $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$.

The embedding dimension $D$ is split into $h$ heads, each with dimension $d = D/h$.

Each head $i$ performs scaled dot-product attention:
$$\text{Attention}_i(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d}}\right) V_i$$

## Proposed Two-Level Partitioning Scheme

### Partitioning Parameters
- $h$: total number of heads
- $d$: dimension per head ($D = h \times d$)
- $n$: number of head partitions
- $m$: number of dimension partitions per head

Derived parameters:
- $h_g = h/n$: heads per group
- $d_s = d/m$: slice dimension per partition

### Step 1: Partition Weight Matrices
Each projection matrix $W \in \mathbb{R}^{D \times D}$ (for Q, K, V) is partitioned into blocks $W^{(i,j)}$ where:
- $i \in [1, n]$ indexes the head group
- $j \in [1, m]$ indexes the intra-head dimension slice

Each block:
$$W^{(i,j)} \in \mathbb{R}^{d_s \cdot h_g \times d_s \cdot h_g}$$

### Step 2: Computation on Each Partition
Each device handling partition $(i,j)$ receives corresponding slices and computes:
$$Q^{(i,j)} = X W_Q^{(i,j)}, \quad K^{(i,j)} = X W_K^{(i,j)}, \quad V^{(i,j)} = X W_V^{(i,j)}$$

Then computes scaled dot-product attention:
$$\text{Attention}^{(i,j)} = \text{softmax}\left(\frac{Q^{(i,j)} (K^{(i,j)})^\top}{\sqrt{d_s}}\right) V^{(i,j)}$$

### Step 3: Aggregation of Results
Two-stage concatenation:
1. **Within head group**: Concatenate dimension slices $j=1,...,m$ along feature dimension
2. **Across head groups**: Concatenate head groups $i=1,...,n$ along head dimension

Final output:
$$\text{Output} = \text{Concat}_{i=1}^n \left( \text{Concat}_{j=1}^m \text{Attention}^{(i,j)} \right)$$

## Communication and Synchronization

### Data Flow
1. **Input distribution**: Each device receives its corresponding input slice for projections
2. **Intra-group communication**: Partial results within head group must be concatenated
3. **Inter-group communication**: Final head groups' outputs concatenated (minimal if placed optimally)

### Communication Efficiency
- Hierarchical partitioning reduces communication overhead vs naive full-dimension splits
- Localized intra-head dimension partitions minimize cross-device synchronization
- Placement optimization can eliminate some inter-group communication

## Implementation Details

### Integration Requirements
- Compatible with existing model parallel frameworks
- Requires custom tensor partitioning and communication primitives
- Supports both training and inference with adapted gradient synchronization

### Configuration Parameters
- Choice of $m$ and $n$ depends on:
  - Hardware topology
  - Network bandwidth
  - Model characteristics (h, d, D)
  - Available device count

### Memory Considerations
- Each device stores only fraction of MHA parameters
- Intermediate activations also distributed across devices
- Memory footprint per device = $O(\frac{D^2}{m \times n})$ for weights + $O(\frac{B \times L \times D}{m \times n})$ for activations

## Algorithm Summary

```
Input: Input tensor X, partition parameters (m, n)
Output: MHA output tensor

// Step 1: Partition setup
h_g = h / n  // heads per group
d_s = d / m  // dimension per slice

// Step 2: Distribute computation
for each device (i,j) in [1..n] × [1..m]:
    Compute Q^(i,j), K^(i,j), V^(i,j) using local weights
    Compute Attention^(i,j) using local attention

// Step 3: Hierarchical aggregation
for each head group i in [1..n]:
    Concatenate dimension slices: Attention^i = Concat_j(Attention^(i,j))

Final output = Concat_i(Attention^i)
```