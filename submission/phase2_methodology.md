# Phase 2: Methodology Extraction

## Multi-Head Attention Foundation
- Input tensor: X ∈ ℝ^(B×L×D) where B=batch size, L=sequence length, D=embedding dimension
- MHA projects X into Q, K, V using weight matrices W_Q, W_K, W_V ∈ ℝ^(D×D)
- D = h × d where h=number of heads, d=dimension per head

## Two-Level Partitioning Scheme

### Parameter Definitions
- h: total number of heads (fixed at 16 in experiments)
- d: dimension per head (fixed at 512 in experiments)
- D: total embedding dimension = h × d = 16 × 512 = 8192
- n: number of head partitions
- m: number of dimension partitions per head
- h_g = h/n: heads per group
- d_s = d/m: slice dimension per partition

### Partitioning Structure
- Total partitions: m × n (16 in experiments, so m×n=16)
- Each partition handles: h_g heads × d_s dimensions

## Weight Matrix Partitioning
- Each projection matrix W ∈ ℝ^(D×D) partitioned into blocks W^(i,j)
- i ∈ [1,n] indexes head group
- j ∈ [1,m] indexes intra-head dimension slice
- Block size: W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g)

## Computation Flow
1. **Input Projection**: Each device (i,j) computes:
   - Q^(i,j) = X W_Q^(i,j)
   - K^(i,j) = X W_K^(i,j)
   - V^(i,j) = X W_V^(i,j)

2. **Attention Computation**: Each device computes:
   - Attention^(i,j) = softmax(Q^(i,j)(K^(i,j))^T/√d_s) V^(i,j)

3. **Aggregation Process**:
   - Step 1: Concatenate dimension slices j=1..m within each head group i
   - Step 2: Concatenate head groups i=1..n along head dimension
   - Final output: Output = Concat_i=1^n(Concat_j=1^m Attention^(i,j))

## Communication Pattern
- **Input Distribution**: Each device receives corresponding input slice for projections
- **Intra-group Communication**: Devices within same head group communicate to concatenate dimension slices
- **Inter-group Communication**: Minimal - head groups concatenated without additional communication if properly placed

## Implementation Specifications
- Compatible with existing model parallel frameworks
- Supports both training and inference modes
- Precision: Mixed precision (FP16) as used in experiments
- Batch size: 1024 (fixed in experiments)
- Device mapping: Direct mapping of m×n partitions to m×n devices

## Partitioning Examples from Experiments
For 16 GPUs (m×n=16):
- Option 1: m=4, n=4 → 4×4=16 partitions
- Option 2: m=2, n=8 → 2×8=16 partitions
- Option 3: m=8, n=2 → 8×2=16 partitions
- Each partition processes: (16/n) heads × (512/m) dimensions