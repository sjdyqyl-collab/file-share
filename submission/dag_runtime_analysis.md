# DAG Runtime Analysis: Two-Level Attention Partitioning

## DAG Structure Overview

The proposed method implements a two-level partitioning scheme for multi-head attention across 16 devices. The DAG consists of the following key operations:

### Matrix Multiplication Operations

#### 1. Input Projection Phase
**Operation**: Input tensor projection to Q, K, V matrices
- **Input matrix X**: ℝ^(B×L×D) where B=1024, L=sequence length, D=8192
- **Weight matrices**: W_Q^(i,j), W_K^(i,j), W_V^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g)
- **Dimensions per partition**: 
  - h_g = 16/n heads per group
  - d_s = 512/m dimensions per segment
  - Block size: d_s·h_g × d_s·h_g = (512/m)·(16/n) × (512/m)·(16/n)

For m=4, n=4 configuration:
- h_g = 4 heads per group
- d_s = 128 dimensions per segment
- Block size: 512 × 512

**Matrix multiplication dimensions**:
- X: ℝ^(1024×L×8192) × W: ℝ^(512×512) → Output: ℝ^(1024×L×512)
- **Count**: 3 operations per partition (Q, K, V projections)
- **Total**: 3 × 16 = 48 matrix multiplications across all partitions

#### 2. Attention Computation Phase
**Operation**: Scaled dot-product attention
- **Q^(i,j)**: ℝ^(1024×L×512) for each partition
- **K^(i,j)^T**: ℝ^(512×L×1024) transposed
- **V^(i,j)**: ℝ^(1024×L×512)

**Matrix multiplication dimensions**:
1. Q × K^T: ℝ^(1024×L×512) × ℝ^(512×L×1024) → ℝ^(1024×L×L)
2. Attention weights × V: ℝ^(1024×L×L) × ℝ^(1024×L×512) → ℝ^(1024×L×512)

**Count**: 2 operations per partition
**Total**: 2 × 16 = 32 matrix multiplications across all partitions

### Longest Path Analysis

The DAG has the following critical path:

1. **Input Distribution**: Broadcast input X to all 16 devices (parallel)
2. **Projection Phase**: 
   - Device (i,j) computes Q^(i,j) = X W_Q^(i,j) [parallel across all devices]
   - Device (i,j) computes K^(i,j) = X W_K^(i,j) [parallel across all devices]
   - Device (i,j) computes V^(i,j) = X W_V^(i,j) [parallel across all devices]
3. **Attention Computation**:
   - Device (i,j) computes Attention^(i,j) = softmax(Q^(i,j)(K^(i,j))^T/√d_s)V^(i,j) [parallel]
4. **Aggregation Phase**:
   - Intra-group concatenation: Concatenate j=1..m within each head group i [parallel across groups]
   - Inter-group concatenation: Concatenate i=1..n head groups [sequential dependency]

**Critical Path Length**: 4 main stages with parallel execution within each stage

### Runtime Calculation

Using the Get_Time function for matrix multiplication timing:

#### Stage 1: Input Projections (Parallel)
- **Operation**: X × W_Q^(i,j)
- **Dimensions**: m=1024×L, k=8192, n=512
- **Time**: Get_Time(1024×L, 8192, 512) for each projection
- **Total for stage**: Get_Time(1024×L, 8192, 512) (parallel execution)

#### Stage 2: Attention Computation (Parallel)
- **Operation 1**: Q × K^T
- **Dimensions**: m=1024×L, k=512, n=L
- **Time**: Get_Time(1024×L, 512, L)

- **Operation 2**: Attention × V
- **Dimensions**: m=1024×L, k=L, n=512
- **Time**: Get_Time(1024×L, L, 512)

#### Stage 3: Intra-group Concatenation (Parallel)
- **Communication overhead**: 0.22ms (dense) as reported
- **No matrix multiplication** - purely concatenation

#### Stage 4: Inter-group Concatenation (Sequential)
- **Communication overhead**: Included in 0.22ms total
- **Final aggregation**: Concatenation across head groups

### Total Runtime Formula

For the proposed method (m×n=16):

```
Total_Runtime = Get_Time(1024×L, 8192, 512) +      // Q projection
                Get_Time(1024×L, 8192, 512) +      // K projection
                Get_Time(1024×L, 8192, 512) +      // V projection
                Get_Time(1024×L, 512, L) +         // Q×K^T
                Get_Time(1024×L, L, 512) +         // Attention×V
                0.22e-3                            // Communication overhead
```

### Comparison with Baseline

**Baseline (TP=8, PP=2)**:
- Uses tensor parallelism (degree 8) + pipeline parallelism (degree 2)
- Different matrix multiplication dimensions due to partitioning
- Reported overhead: 0.35ms (dense)

**Key Advantages of Proposed Method**:
1. **Reduced communication**: 0.22ms vs 0.35ms (37% reduction)
2. **Better load balancing**: Equal work across all 16 devices
3. **Scalability**: Can scale beyond head count limitations
4. **Parallel efficiency**: All 16 devices work simultaneously on equal-sized partitions

### Matrix Multiplication Summary

| Operation | Count per Device | Dimensions | Parallelism |
|-----------|------------------|------------|-------------|
| X×W_Q     | 1                | (1024×L, 8192, 512) | 16-way |
| X×W_K     | 1                | (1024×L, 8192, 512) | 16-way |
| X×W_V     | 1                | (1024×L, 8192, 512) | 16-way |
| Q×K^T     | 1                | (1024×L, 512, L) | 16-way |
| Attention×V | 1              | (1024×L, L, 512) | 16-way |

**Total matrix multiplications in DAG**: 80 (5 per device × 16 devices)
**Longest path**: 5 sequential matrix multiplications plus communication overhead
**Critical path runtime**: Sum of 5 Get_Time calls plus 0.22ms communication

### Performance Characteristics

The DAG achieves optimal performance when:
- m×n = 16 (full GPU utilization)
- L = 2048 (typical sequence length)
- Batch size = 1024 (as in experiments)

This configuration yields the reported 1.58M tokens/sec throughput for dense models.