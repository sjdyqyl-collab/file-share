# Two-Level Attention Partitioning for Large-Scale Transformers - Concise Version

### Abstract
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Introduction
Transformer architectures with multi-head attention (MHA) face scaling challenges as model sizes grow exponentially. Traditional MHA parallelization splits attention heads across devices, but this approach is limited by the fixed number of heads and creates communication bottlenecks when devices exceed heads. We introduce a two-level partitioning strategy that segments both heads and internal dimensions, enabling flexible scaling to m×n devices.

## Method

### Two-Level Partitioning Scheme
Our method partitions MHA along two dimensions:
1. **Head Dimension Partitioning**: Divide h heads into n groups (h_g = h/n heads per group)
2. **Intra-Head Dimension Partitioning**: Split each head's feature dimension d into m segments (d_s = d/m per segment)

This creates m×n total partitions, each corresponding to a unique (head_group, dimension_slice) pair.

### Implementation Details
Given input X ∈ ℝ^(B×L×D):
- Weight matrices W_Q, W_K, W_V ∈ ℝ^(D×D) partitioned into blocks W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g)
- Each device computes: Attention^(i,j) = softmax(Q^(i,j)(K^(i,j))^T/√d_s)V^(i,j)
- Results concatenated in two stages: dimension slices first, then head groups

### Advantages
- **Scalability**: Supports m×n devices regardless of head count
- **Load Balancing**: Even distribution across heads and dimensions
- **Memory Efficiency**: Each device stores 1/(m×n) of parameters
- **Communication Efficiency**: Reduced synchronization bandwidth

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs, FP16 precision
- **Models**: 2-layer Dense Transformer, 2-layer MoE Transformer (4 experts/layer)
- **Fixed**: batch=1024, heads=16, head_dim=512, MLP_hidden=32768

### Baseline
Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2) across 16 GPUs

### Results
| Model | Method | TPS (tokens/sec) | TPOT (ms) |
|-------|--------|------------------|-----------|
| Dense | Baseline | 1,200,000 | 0.35 |
| Dense | Proposed | 1,580,000 | 0.22 |
| MoE   | Baseline | 850,000  | 0.45 |
| MoE   | Proposed | 1,150,000 | 0.30 |

### Performance Gains
- **Dense model**: +31.7% throughput, -37.1% overhead
- **MoE model**: +35.3% throughput, -33.3% overhead

## Conclusion
Our two-level partitioning method enables efficient deployment of MHA across m×n devices by combining head-wise and dimension-wise partitioning. Experiments demonstrate substantial improvements in throughput (up to 35%) and reduced communication overhead (over 30%) compared to traditional tensor+pipeline parallelism, validating the approach for large-scale transformer deployment.