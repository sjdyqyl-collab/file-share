# Helix: Two-Level Attention Partitioning - Key Points

## Abstract (Retained in full)
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Key Contributions
1. **Two-level partitioning method** for MHA layers
2. **Head-level partitioning**: h heads divided into n groups (h_g = h/n heads per group)
3. **Intra-head dimension partitioning**: Each head's dimension d sliced into m segments (d_s = d/m per segment)
4. **Total partitions**: m × n partitions mapped to m × n devices
5. **Improved scalability** beyond traditional head-wise splitting
6. **Reduced communication overhead** through hierarchical partitioning

## Critical Parameters
- Model: 2-layer Dense Transformer
- Hardware: 16 NVIDIA H100 GPUs
- Batch size: 1024
- Heads: 16
- Head dimension: 512
- MLP hidden size: 32768
- Precision: FP16

## Performance Results
- **Baseline (TP=8, PP=2)**: 1,200,000 TPS, 0.35ms TPOT
- **Proposed (m×n=16)**: 1,580,000 TPS, 0.22ms TPOT
- **Improvements**: 31.7% throughput increase, 37.1% overhead reduction