# Phase 1: Keypoints Extraction - Two-Level Attention Partitioning for Large-Scale Transformers

## Abstract (Original)
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Key Problem
- Traditional MHA parallelization only splits attention heads across devices
- Limited by fixed number of heads (h)
- Cannot fully exploit hardware parallelism when devices > heads
- Leads to suboptimal utilization and communication bottlenecks

## Key Innovation
- **Two-level partitioning scheme** that combines:
  1. **Head-level partitioning**: Split h heads into n groups (h/n heads per group)
  2. **Intra-head dimension partitioning**: Split each head's feature dimension d into m segments (d/m per segment)
- Results in m × n total partitions that can be mapped to m × n devices

## Key Technical Details
- Input tensor: X ∈ ℝ^(B×L×D) where B=batch, L=sequence, D=embedding
- h heads, each with dimension d = D/h
- Partition parameters:
  - h_g = h/n: heads per group
  - d_s = d/m: slice dimension per partition
- Weight matrices W_Q, W_K, W_V ∈ ℝ^(D×D) partitioned into blocks W^(i,j)
- Each partition computes: Attention^(i,j) = softmax(Q^(i,j)(K^(i,j))ᵀ/√d_s)V^(i,j)
- Results concatenated in two stages: dimension slices first, then head groups

## Key Advantages
- **Scalability**: Supports m×n devices, exceeding head-wise splitting limits
- **Load Balancing**: Even workload distribution across both heads and dimensions
- **Memory Efficiency**: Each device stores only fraction of parameters/activations
- **Communication Efficiency**: Localized partitions reduce synchronization bandwidth

## Key Experimental Results
- **Setup**: 16 NVIDIA H100 GPUs, FP16, batch=1024, heads=16, head_dim=512, MLP_hidden=32768
- **Models**: 2-layer Dense Transformer, 2-layer MoE Transformer (4 experts/layer)
- **Baseline**: Tensor Parallelism=8 + Pipeline Parallelism=2
- **Results**:
  - Dense model: 31.7% throughput improvement (1.2M→1.58M tokens/sec), 37.1% overhead reduction
  - MoE model: 35.3% throughput improvement (850K→1.15M tokens/sec), 33.3% overhead reduction

## Key Deployment Mapping
- m×n=16 partitions mapped to 16 devices (m=4, n=4 configuration implied)
- Fully utilizes all 16 GPUs vs baseline's TP8+PP2 approach