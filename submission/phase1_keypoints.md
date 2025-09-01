# Phase One: Key Points Extraction

## Abstract
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Core Problem
- Transformer models with multi-head attention (MHA) need efficient distributed deployment
- Traditional head-wise splitting is limited by fixed number of heads (h)
- Cannot fully utilize hardware when devices > heads
- Communication bottlenecks and suboptimal utilization in large clusters

## Key Innovation
- **Two-level partitioning method** for MHA layers
- **Level 1**: Split h heads into n groups (each group has h/n heads)
- **Level 2**: Split each head's feature dimension d into m segments (each segment has d/m dimensions)
- **Result**: m × n partitions that can be mapped to m × n devices

## Technical Specifications
- Input tensor: X ∈ ℝ^(B×L×D) where B=batch size, L=sequence length, D=embedding dimension
- h = number of heads
- d = dimension per head (D = h × d)
- n = head partitions
- m = dimension partitions per head
- h_g = h/n (heads per group)
- d_s = d/m (slice dimension per partition)

## Key Benefits
1. **Scalability**: Supports m×n devices, exceeding head-wise splitting limits
2. **Load Balancing**: Even workload distribution across heads and dimensions
3. **Reduced Memory**: Each device stores fraction of parameters and activations
4. **Communication Efficiency**: Localized partitions reduce synchronization bandwidth

## Experimental Validation
- **Setup**: 16 NVIDIA H100 GPUs, FP16 precision
- **Models**: 2-layer Dense Transformer, 2-layer MoE Transformer (4 experts/layer)
- **Fixed Parameters**: Batch size=1024, heads=16, head dimension=512, MLP hidden size=32768
- **Results**:
  - Dense model: 31.7% throughput improvement (1.2M→1.58M tokens/sec), 37.1% overhead reduction
  - MoE model: 35.3% throughput improvement (850K→1.15M tokens/sec), 33.3% overhead reduction

## Baseline Comparison
- **Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2) on 16 GPUs
- **Proposed**: m×n=16 partitions (likely m=4, n=4 based on 16 total partitions)