# Phase 2: Methodology Extraction

## Two-Level Partitioning Method for Multi-Head Attention

### Overview
The proposed method partitions MHA computation along two dimensions simultaneously:
1. **Head dimension partitioning** - splits h heads into n groups
2. **Intra-head dimension partitioning** - splits each head's d dimensions into m segments

### Mathematical Formulation

#### Input Specifications
- Input tensor: X ∈ ℝ^(B×L×D)
- B: batch size
- L: sequence length  
- D: embedding dimension
- h: number of attention heads
- d: dimension per head (D = h × d)

#### Partitioning Parameters
- n: number of head partitions (groups)
- m: number of dimension partitions per head
- h_g = h/n: heads per group (must be integer)
- d_s = d/m: dimension slice per partition (must be integer)
- Total partitions: m × n

#### Weight Matrix Partitioning
For each projection matrix (W_Q, W_K, W_V ∈ ℝ^(D×D)):
- Partition into blocks W^(i,j) where:
  - i ∈ [1,n]: head group index
  - j ∈ [1,m]: dimension slice index
- Each block: W^(i,j) ∈ ℝ^(d_s×h_g × d_s×h_g)

#### Computation Flow
1. **Projection**: Each device (i,j) computes:
   - Q^(i,j) = X W_Q^(i,j)
   - K^(i,j) = X W_K^(i,j)  
   - V^(i,j) = X W_V^(i,j)

2. **Attention Computation**: Each device computes:
   - Attention^(i,j) = softmax(Q^(i,j) (K^(i,j))^T / √d_s) V^(i,j)

3. **Aggregation**:
   - **Step 1**: Concatenate dimension slices within each head group
     - For each group i: Concat_j=1^m Attention^(i,j)
   - **Step 2**: Concatenate head groups
     - Final output: Concat_i=1^n (Concat_j=1^m Attention^(i,j))

### Communication Pattern
- **Input distribution**: Each device receives corresponding input slice for projections
- **Intra-group communication**: Devices within the same head group communicate to concatenate dimension slices
- **Final concatenation**: Head group outputs concatenated without additional communication if properly placed

### Implementation Requirements
- Must support custom tensor partitioning
- Requires communication primitives for intra-group concatenation
- Compatible with existing model parallel frameworks
- Supports both training and inference modes
- Choice of m and n depends on hardware topology and network bandwidth

### Memory and Computation Distribution
- Each device stores: 1/(m×n) of total MHA parameters
- Each device computes: attention for d_s dimensions across h_g heads
- Memory footprint per device: O(B×L×d_s×h_g) for activations
- Computation per device: O(B×L²×d_s×h_g) for attention