# Helix: Two-Level Attention Partitioning for Large-Scale Transformers

### Abstract
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Introduction
Transformer architectures face scaling challenges as model sizes grow exponentially. Traditional MHA parallelization splits attention heads across devices, but this approach is limited by the fixed number of heads and creates communication bottlenecks when devices exceed heads. We introduce a two-level partitioning strategy that extends beyond head-wise splitting by segmenting each attention head's internal dimension, enabling flexible scaling and better hardware utilization.

## Method

### Two-Level Partitioning Scheme
Given input tensor X ∈ ℝ^(B×L×D) where B=batch size, L=sequence length, D=embedding dimension:

**Level 1**: Split h heads into n groups, each containing h_g = h/n heads
**Level 2**: Split each head's dimension d into m segments, each with d_s = d/m dimensions

**Result**: m×n partitions mapped to m×n devices

### Weight Matrix Partitioning
Each projection matrix W ∈ ℝ^(D×D) (Q, K, V) is partitioned into blocks W^(i,j) where:
- i ∈ [1,n] indexes head group
- j ∈ [1,m] indexes dimension slice
- W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g)

### Computation Flow
1. **Input Distribution**: Each device receives corresponding input slice
2. **Local Projection**: Q^(i,j) = X W_Q^(i,j), K^(i,j) = X W_K^(i,j), V^(i,j) = X W_V^(i,j)
3. **Attention Computation**: softmax(Q^(i,j)(K^(i,j))^T/√d_s) V^(i,j)
4. **Aggregation**: Concatenate dimension slices within groups, then concatenate head groups

### Communication Pattern
- Hierarchical communication reduces overhead
- Intra-group concatenation for dimension slices
- Inter-group concatenation for head outputs
- Localized partitions minimize synchronization bandwidth

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs, FP16 precision
- **Models**: 2-layer Dense Transformer, 2-layer MoE Transformer (4 experts/layer)
- **Fixed Parameters**: Batch size=1024, heads=16, head dimension=512, MLP hidden=32768

### Results
| Model | Method | TPS (tokens/sec) | TPOT (ms) |
|-------|--------|------------------|-----------|
| Dense | Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| Dense | Proposed (m×n=16) | 1,580,000 | 0.22 |
| MoE | Baseline (TP=8, PP=2) | 850,000 | 0.45 |
| MoE | Proposed (m×n=16) | 1,150,000 | 0.30 |

### Performance Gains
- **Dense model**: +31.7% throughput, -37.1% overhead
- **MoE model**: +35.3% throughput, -33.3% overhead

## Conclusion
Our two-level partitioning method enables deployment across m×n devices by combining head-wise and dimension-wise splitting. Experiments demonstrate substantial improvements in throughput (up to 35%) and communication overhead reduction (over 30%) compared to tensor and pipeline parallelism baselines, validating this approach for large-scale transformer deployment.

## Deployment Configuration
For implementation details, see `deployment_config.json` which provides complete specifications for both proposed method and baseline configurations, including tensor dimensions, weight partitioning strategies, communication patterns, and performance targets.