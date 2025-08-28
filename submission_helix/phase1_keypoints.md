# Phase 1: Key Points Extraction

## Problem Statement
- Transformer models with multi-head attention (MHA) are growing exponentially in size
- Traditional MHA parallelization only splits attention heads across devices
- This approach leads to suboptimal utilization and communication bottlenecks when the number of available devices exceeds the number of heads
- Need for efficient distributed deployment of very large models across numerous devices

## Proposed Solution
- **Two-level partitioning method** for MHA layers
- **First level**: Split attention heads into *n* groups (head-wise partitioning)
- **Second level**: Split each head's feature dimension into *m* segments (intra-head dimension partitioning)
- Results in *m × n* total partitions that can be mapped to *m × n* devices
- Enables deployment beyond traditional head-wise splitting limits

## Key Contributions
1. **Novel partitioning strategy** that combines head-wise and dimension-wise splitting
2. **Improved scalability** by supporting deployment on *m × n* devices regardless of number of heads
3. **Better load balancing** through fine-grained workload distribution
4. **Reduced communication overhead** through localized computations
5. **Reduced memory footprint** as each device only stores a fraction of parameters

## Technical Details
- Input tensor: X ∈ ℝ^(B×L×D) where B=batch size, L=sequence length, D=embedding dimension
- Total heads: h, dimension per head: d = D/h
- Head groups: h_g = h/n heads per group
- Dimension slices: d_s = d/m per partition
- Each partition handles: d_s × h_g dimensions

## Experimental Results
- Tested on 16 NVIDIA H100 GPUs
- Models: 2-layer Dense Transformer and 2-layer MoE Transformer (4 experts/layer)
- Metrics: Throughput (TPS) and Time Per Output Token (TPOT)
- **Dense model improvements**: 31.7% throughput increase (1.2M→1.58M tokens/sec), 37.1% overhead reduction
- **MoE model improvements**: 35.3% throughput increase (850K→1.15M tokens/sec), 33.3% overhead reduction
- Baseline: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- Proposed: m×n=16 partitions (fully utilizing 16 GPUs)