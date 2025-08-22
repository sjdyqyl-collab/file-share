# Two-Level Attention Partitioning for Large-Scale Transformers: A Concise Version

### Abstract
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Introduction
Transformer architectures with multi-head attention (MHA) are growing exponentially, requiring efficient distributed deployment. Traditional MHA parallelization only splits attention heads across devices, leading to suboptimal utilization when available devices exceed head count. We introduce a two-level partitioning strategy that extends beyond conventional head-wise splitting by segmenting each attention head's internal dimension, enabling flexible scaling and reduced communication overhead.

## Method

### Two-Level Partitioning Scheme
- **Head-level partitioning**: Divide h heads into n groups (h_g = h/n heads per group)
- **Dimension-level partitioning**: Slice each head's dimension d into m segments (d_s = d/m per segment)
- **Total partitions**: m × n partitions mapped to m × n devices

### Mathematical Formulation
- **Input**: X ∈ ℝ^(B×L×D) where B=batch size, L=sequence length, D=embedding dimension
- **Weight partitioning**: Each W_Q, W_K, W_V ∈ ℝ^(D×D) split into blocks W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g)
- **Per-device computation**: Each device computes Attention^(i,j) = softmax(Q^(i,j)(K^(i,j))^T/√d_s)V^(i,j)
- **Aggregation**: Hierarchical concatenation - first within head groups, then across groups

### Communication Pattern
- Input distribution to devices based on partition (i,j)
- Intra-group concatenation within head groups
- Inter-group concatenation across head groups
- Hierarchical reduction minimizes communication overhead

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs with FP16 precision
- **Models**: 4-layer Dense Transformer and 4-layer MoE Transformer (8 experts/layer)
- **Fixed parameters**: Batch size=1024, heads=16, dim/head=512, MLP hidden=32768
- **Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- **Proposed**: m×n=16 partitions (2×8 or 4×4 configuration)

### Results
| Model | Method | TPS (tokens/sec) | TPOT (ms) | Improvement |
|-------|--------|------------------|-----------|-------------|
| Dense | Baseline | 1,200,000 | 0.35 | - |
| Dense | Proposed | 1,580,000 | 0.22 | +31.7% TPS, -37.1% overhead |
| MoE | Baseline | 850,000 | 0.45 | - |
| MoE | Proposed | 1,150,000 | 0.30 | +35.3% TPS, -33.3% overhead |

### Key Findings
- 31.7% throughput improvement for dense models
- 35.3% throughput improvement for MoE models
- 33-37% reduction in communication overhead
- Full hardware utilization across 16 GPUs
- Compatible with existing model parallel frameworks

## Conclusion
Our two-level partitioning method enables efficient deployment of large transformer models by combining head-wise and dimension-wise partitioning. The approach achieves substantial performance improvements while maintaining compatibility with existing distributed training frameworks, offering a promising direction for scaling transformer architectures.