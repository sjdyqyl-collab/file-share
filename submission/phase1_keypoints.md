# Helix: Two-Level Attention Partitioning for Large-Scale Transformers - Key Points

## Abstract (Retained in full)
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Key Points

### Problem Statement
- Traditional MHA parallelization only splits attention heads across devices
- Limited by fixed number of heads (h) when scaling to large clusters
- Suboptimal utilization and communication bottlenecks when devices > heads

### Proposed Solution
- **Two-level partitioning scheme**:
  1. Head dimension partitioning: h heads → n groups (h/n heads per group)
  2. Intra-head dimension partitioning: d dimension per head → m segments (d/m per segment)
- Results in m×n total partitions for m×n devices

### Technical Innovation
- Fine-grained distribution beyond conventional head-wise splitting
- Enables flexible scaling beyond traditional limits
- Reduces memory footprint per device
- Improves load balancing and communication efficiency

### Experimental Validation
- **Setup**: 16 NVIDIA H100 GPUs, FP16 precision
- **Models**: 4-layer Dense Transformer and 4-layer MoE Transformer
- **Results**:
  - Dense model: 31.7% throughput improvement (1.2M → 1.58M tokens/sec)
  - MoE model: 35.3% throughput improvement (850K → 1.15M tokens/sec)
  - Communication overhead reduced by 33-37%

### Key Advantages
- Scalability: Supports m×n devices beyond head count limitations
- Load balancing: Even division of both heads and dimensions
- Memory efficiency: Each device stores only fraction of parameters
- Communication efficiency: Localized partitions reduce synchronization

### Deployment Requirements
- Can integrate with existing model parallel frameworks
- Supports both training and inference
- Choice of m and n depends on hardware topology and bandwidth