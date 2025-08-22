# Helix: Two-Level Attention Partitioning for Large-Scale Transformers - Phase 1: Key Points

## Abstract
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Key Points

### Problem Statement
- Transformer models are growing exponentially in size
- Traditional MHA parallelization only splits attention heads across devices
- This approach is limited by the fixed number of heads and leads to suboptimal utilization when devices > heads
- Communication bottlenecks occur with naive head-wise splitting

### Novel Contribution
- **Two-level partitioning scheme** that goes beyond conventional head-wise splitting
- **First level**: Split h heads into n groups (h/n heads per group)
- **Second level**: Split each head's feature dimension d into m segments (d/m per segment)
- Results in **m × n total partitions** that can be mapped to m × n devices
- Enables deployment on more devices than the number of attention heads

### Technical Innovation
- **Dual granularity**: Head-level + intra-head dimension-level partitioning
- **Hierarchical aggregation**: First concatenate dimension slices within head groups, then concatenate head groups
- **Reduced communication**: Localized intra-head dimension partitions minimize cross-device synchronization
- **Better load balancing**: Even distribution of both head count and feature dimension

### Performance Results
- **31.7% throughput improvement** for dense transformer (1.2M → 1.58M tokens/sec)
- **35.3% throughput improvement** for MoE transformer (850K → 1.15M tokens/sec)
- **37.1% reduction in communication overhead** for dense model
- **33.3% reduction in communication overhead** for MoE model
- Tested on 16 NVIDIA H100 GPUs with batch size 1024

### Key Benefits
- **Scalability**: Supports deployment on m × n devices, exceeding traditional head-wise limits
- **Memory efficiency**: Each device stores only a fraction of parameters and activations
- **Hardware utilization**: Better exploitation of available devices
- **Flexibility**: Can be integrated with existing model parallel frameworks

### Implementation Details
- Compatible with both training and inference
- Works with mixed precision (FP16)
- Requires customization of tensor partitioning and communication primitives
- Choice of m and n depends on hardware topology and network bandwidth