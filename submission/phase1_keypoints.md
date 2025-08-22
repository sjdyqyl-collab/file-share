# Phase 1: Key Points Extraction - Helix: Two-Level Attention Partitioning for Distributed Transformer Models

## Abstract (Retained as-is)
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Key Problem
- Transformer models with multi-head attention (MHA) are growing exponentially in size
- Traditional MHA parallelization only splits attention heads across devices
- This approach is limited when number of devices exceeds number of heads
- Leads to suboptimal utilization and communication bottlenecks

## Key Innovation
- **Two-level partitioning method** that goes beyond conventional head-wise splitting
- **First level**: Split h heads into n groups (each group has h/n heads)
- **Second level**: Split each head's feature dimension d into m segments (each segment has d/m dimensions)
- **Total partitions**: m × n partitions that can be mapped to m × n devices

## Key Benefits
- **Scalability**: Supports deployment on m × n devices, exceeding traditional head-wise splitting limits
- **Load Balancing**: Even workload distribution across both head count and feature dimension
- **Reduced Memory**: Each device stores only fraction of MHA parameters and activations
- **Communication Efficiency**: Localized intra-head dimension partitions reduce cross-device synchronization

## Key Technical Details
- Each partition computes attention for subset of heads and slice of dimensions
- Results aggregated through hierarchical concatenation
- Compatible with existing model parallel frameworks
- Supports both training and inference
- Choice of m and n depends on hardware topology and network bandwidth

## Key Results
- **31.7% throughput improvement** for 4-layer Dense Transformer (1.2M → 1.58M tokens/sec)
- **35.3% throughput improvement** for 4-layer MoE Transformer (850K → 1.15M tokens/sec)
- **37.1% overhead reduction** for Dense model (0.35 → 0.22 ms)
- **33.3% overhead reduction** for MoE model (0.45 → 0.30 ms)
- Experiments conducted on 16 NVIDIA H100 GPUs with m×n=16 configuration

## Key Future Directions
- Extend partitioning scheme to training scenarios
- Investigate adaptive partitioning strategies based on model characteristics and hardware topology