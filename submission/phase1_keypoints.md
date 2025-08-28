# Phase 1: Keypoints Extraction - Helix: Two-Level Attention Partitioning for Large-Scale Transformers

## Problem Statement
- Transformer models with multi-head attention (MHA) are growing exponentially in size
- Traditional MHA parallelization only splits attention heads across devices
- This approach leads to suboptimal utilization and communication bottlenecks when devices > heads
- Need for better parallelization strategy for very large clusters

## Proposed Solution
- **Two-level partitioning method** for MHA layers
- Combines head-level partitioning with intra-head dimension partitioning
- Creates m×n partitions (m = dimension slices, n = head groups)
- Enables deployment on m×n devices beyond traditional limits

## Key Innovations
1. **Dual-level slicing**: Head groups (n) + dimension segments (m)
2. **Fine-grained distribution**: Each partition handles (head group, dimension slice) pair
3. **Improved scalability**: Supports deployment on m×n devices
4. **Better load balancing**: Even division of both heads and dimensions
5. **Reduced communication**: Localized intra-head partitions reduce synchronization

## Technical Details
- Input tensor: X ∈ ℝ^(B×L×D) where B=batch, L=sequence, D=embedding
- h heads, each with dimension d = D/h
- Partition parameters: n head groups, m dimension slices per head
- Each partition handles h_g = h/n heads and d_s = d/m dimensions

## Experimental Results
- **Setup**: 16 NVIDIA H100 GPUs, FP16, batch size 1024
- **Models tested**: 2-layer Dense Transformer, 2-layer MoE Transformer (4 experts/layer)
- **Fixed parameters**: 16 heads, 512 head dimension, 32768 MLP hidden size
- **Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)

## Performance Improvements
- **Dense model**: 31.7% throughput increase (1.2M → 1.58M tokens/sec), 37.1% overhead reduction
- **MoE model**: 35.3% throughput increase (850K → 1.15M tokens/sec), 33.3% overhead reduction
- **TPOT reduction**: 0.35ms → 0.22ms (dense), 0.45ms → 0.30ms (MoE)