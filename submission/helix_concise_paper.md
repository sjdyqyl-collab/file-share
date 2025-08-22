# Helix: Two-Level Attention Partitioning for Large-Scale Transformers (Concise Version)

## Abstract
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Introduction
Transformer models are growing exponentially, requiring efficient distribution across hardware. Traditional MHA parallelization splits attention heads across devices, but this is limited by the fixed number of heads and creates bottlenecks when devices > heads. We introduce a two-level partitioning strategy that segments both heads and intra-head dimensions, enabling flexible scaling beyond traditional limits.

## Method

### Two-Level Partitioning Scheme
- **Level 1**: Split h heads into n groups (h/n heads per group)
- **Level 2**: Split each head's feature dimension d into m segments (d/m per segment)
- **Result**: m × n partitions mapped to m × n devices

### Mathematical Formulation
- Input: X ∈ ℝ^(B×L×D), where D = h×d
- Partition parameters: h_g = h/n (heads per group), d_s = d/m (slice dimension)
- Weight matrices W_Q, W_K, W_V ∈ ℝ^(D×D) partitioned into blocks W^(i,j) ∈ ℝ^(d_s×h_g × d_s×h_g)

### Computation Flow
1. **Projection**: Each device computes Q^(i,j), K^(i,j), V^(i,j) = X W^(i,j)
2. **Attention**: Compute Attention^(i,j) = softmax(Q^(i,j)(K^(i,j))^T/√d_s)V^(i,j)
3. **Aggregation**: Hierarchical concatenation - first within head groups, then across groups
4. **Output**: Final output = Concat_{i=1}^n(Concat_{j=1}^m Attention^(i,j))

### Communication Pattern
- All-gather within head groups for dimension slice concatenation
- Concatenation across head groups
- Hierarchical aggregation reduces communication overhead vs. naive splits

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Models**: 4-layer Dense Transformer, 4-layer MoE Transformer (8 experts/layer)
- **Configuration**: 16 heads, 512 dim/head, batch size 1024, FP16 precision
- **Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- **Proposed**: m×n=16 partitions (4×4 configuration)

### Results
| Model | Method | TPS (tokens/sec) | TPOT (ms) |
|-------|--------|------------------|-----------|
| Dense | Baseline | 1,200,000 | 0.35 |
| Dense | Proposed | 1,580,000 | 0.22 |
| MoE   | Baseline | 850,000  | 0.45 |
| MoE   | Proposed | 1,150,000 | 0.30 |

### Key Findings
- **31.7% throughput improvement** for dense transformer
- **35.3% throughput improvement** for MoE transformer  
- **37.1% communication overhead reduction** for dense model
- **33.3% communication overhead reduction** for MoE model
- Full 16/16 GPU utilization achieved

## DAG Generation Specification

### Input Parameters
- Batch size: 1024
- Hidden dimension: 8192 (16×512)
- Sequence length: L (variable)
- Partition count: 16 (m=4, n=4)
- Slice dimension: 128 (512/4)

### Computation Graph
```
Level 1: Input Broadcast
├── All 16 GPUs receive full input X ∈ ℝ^(1024×L×8192)

Level 2: Parallel Projection
├── GPU 0: W_Q^(0,0), W_K^(0,0), W_V^(0,0) → Q^(0,0), K^(0,0), V^(0,0)
├── GPU 1: W_Q^(0,1), W_K^(0,1), W_V^(0,1) → Q^(0,1), K^(0,1), V^(0,1)
├── ...
└── GPU 15: W_Q^(3,3), W_K^(3,3), W_V^(3,3) → Q^(3,3), K^(3,3), V^(3,3)

Level 3: Parallel Attention
├── GPU 0: softmax(Q^(0,0)(K^(0,0))^T/√128)V^(0,0)
├── ...
└── GPU 15: softmax(Q^(3,3)(K^(3,3))^T/√128)V^(3,3)

Level 4: Intra-group Concatenation
├── Group 0 (GPUs 0-3): Concatenate 4 dimension slices
├── Group 1 (GPUs 4-7): Concatenate 4 dimension slices
├── Group 2 (GPUs 8-11): Concatenate 4 dimension slices
└── Group 3 (GPUs 12-15): Concatenate 4 dimension slices

Level 5: Inter-group Concatenation
├── Final output: Concatenate 4 head group outputs
```

### Communication Requirements
- **All-gather operations**: 4× within head groups (4 GPUs each)
- **Concatenation**: 1× across head groups
- **Data transfer**: 1/16 → 1/4 → full tensor reconstruction
- **Synchronization**: After each concatenation level

## Conclusion
The two-level partitioning method enables deployment of MHA computations across m×n devices, achieving 31-35% throughput improvements and 33-37% communication overhead reductions. This approach provides a scalable pathway for efficient distributed inference of large transformer models, overcoming traditional head-count limitations through combined head-wise and dimension-wise partitioning.