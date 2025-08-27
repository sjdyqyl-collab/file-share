# Phase 1: Keypoints Extraction

## Problem Statement
- Transformer models with multi-head attention (MHA) face scaling challenges when distributing across many devices
- Traditional head-wise partitioning is limited by the fixed number of attention heads (h)
- When device count exceeds head count, conventional methods lead to suboptimal utilization and communication bottlenecks

## Key Contribution
- Novel two-level partitioning method for MHA that combines:
  1. Head-level partitioning: splitting h heads into n groups
  2. Intra-head dimension partitioning: splitting each head's feature dimension d into m segments
- Results in m×n total partitions that can be mapped to m×n devices

## Technical Innovation
- Dual-level slicing: head groups (n) × dimension slices (m)
- Each partition handles (h/n) heads × (d/m) dimensions
- Enables deployment on device counts exceeding the number of attention heads

## Key Benefits
- **Scalability**: Supports m×n devices beyond head count limit
- **Load Balancing**: Even distribution across both heads and dimensions
- **Memory Efficiency**: Each device stores fraction of parameters and activations
- **Communication Efficiency**: Reduced synchronization bandwidth through localized partitioning

## Experimental Results
- Tested on 16 NVIDIA H100 GPUs
- Compared against baseline: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- Dense model: 31.7% throughput improvement (1.2M → 1.58M tokens/sec)
- MoE model: 35.3% throughput improvement (850K → 1.15M tokens/sec)
- Communication overhead reduced by 33-37%

## Model Specifications Used
- 2-layer Dense Transformer
- 2-layer MoE Transformer (4 experts per layer)
- Fixed parameters: 16 heads, 512 dimensions per head, batch size 1024, hidden size 32768
- Mixed precision (FP16) for all experiments