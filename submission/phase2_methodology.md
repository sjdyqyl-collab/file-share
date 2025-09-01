# Phase Two: Methodology Extraction

## Multi-Head Attention Background
Given input tensor X ∈ ℝ^(B×L×D) where:
- B = batch size
- L = sequence length  
- D = embedding dimension

MHA layer projects X into query, key, and value tensors:
```
Q, K, V = XW_Q, XW_K, XW_V
```
where W_Q, W_K, W_V ∈ ℝ^(D×D)

Each head i performs scaled dot-product attention:
```
Attention_i(Q_i, K_i, V_i) = softmax(Q_i K_i^T / sqrt(d)) V_i
```
where d = D/h (dimension per head)

## Two-Level Partitioning Scheme

### Level 1: Head Dimension Partitioning
- Split total h heads into n groups
- Each group contains h_g = h/n heads

### Level 2: Intra-Head Dimension Partitioning  
- Split each head's feature dimension d into m segments
- Each segment has d_s = d/m dimensions

### Result
- Total partitions: m × n
- Each partition corresponds to (head group, dimension slice) pair
- Can be mapped to m × n devices

## Weight Matrix Partitioning

### Projection Matrices
Each projection matrix W ∈ ℝ^(D×D) (for Q, K, V) is partitioned into blocks W^(i,j) where:
- i ∈ [1,n] indexes head group
- j ∈ [1,m] indexes intra-head dimension slice
- W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g)

### Partition Structure
- Output dimension: split into h heads
- Input/output dimension of each head: split into m slices
- Each block handles portion of input/output feature spaces

## Computation on Each Partition

### Device Assignment
Each device handling partition (i,j) receives corresponding slices:
```
Q^(i,j) = X W_Q^(i,j)
K^(i,j) = X W_K^(i,j)  
V^(i,j) = X W_V^(i,j)
```

### Local Attention Computation
Each device computes scaled dot-product attention for its slice:
```
Attention^(i,j) = softmax(Q^(i,j) (K^(i,j))^T / sqrt(d_s)) V^(i,j)
```

## Aggregation of Results

### Step 1: Dimension Concatenation
- Concatenate dimension slices j=1,...,m within each head group i
- Reconstruct full head outputs along feature dimension

### Step 2: Head Concatenation  
- Concatenate outputs from all head groups i=1,...,n
- Reconstruct full MHA output along head dimension

### Final Output
```
Output = Concat_{i=1}^n (Concat_{j=1}^m Attention^(i,j))
```
Output matches original MHA layer dimension

## Communication and Synchronization

### Required Communications
1. **Input Distribution**: Each device receives corresponding input slice for projections
2. **Intra-group Concatenation**: Partial results within head group must be concatenated
3. **Final Concatenation**: Head groups' outputs concatenated (no additional communication if placed accordingly)

### Communication Efficiency
- Hierarchical partitioning reduces overhead vs naive full-dimension splits
- Localized intra-head dimension partitions minimize cross-device synchronization

## Implementation Details

### Integration
- Compatible with existing model parallel frameworks
- Requires custom tensor partitioning and communication primitives
- Supports both training and inference with adapted gradient synchronization

### Parameter Selection
- Choice of m and n depends on:
  - Hardware topology
  - Network bandwidth considerations
  - Total number of available devices

### Memory Benefits
- Each device stores fraction of MHA parameters
- Reduced intermediate activation storage
- Better memory distribution across devices