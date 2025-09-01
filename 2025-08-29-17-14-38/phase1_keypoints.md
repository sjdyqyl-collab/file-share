# Key Points of Helix: Two-Level Attention Partitioning for Large-Scale Transformers

## Original Abstract
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Key Points

### Problem Statement
- Traditional MHA parallelization splits attention heads across devices, limited by the fixed number of heads
- This approach leads to suboptimal utilization and communication bottlenecks when devices > heads
- Need for flexible scaling beyond traditional head-wise splitting

### Proposed Solution
- **Two-level partitioning method** for MHA in transformer models
- **First level**: Split h heads into n groups (h/n heads per group)
- **Second level**: Split each head's feature dimension d into m segments (d/m per segment)
- **Result**: m × n total partitions that can be mapped to m × n devices

### Technical Details
- Input tensor: X ∈ ℝ^(B×L×D) where B=batch, L=sequence, D=embedding
- h = number of heads, d = dimension per head, D = h × d
- Each partition handles: d_s × h_g = (d/m) × (h/n) dimensions
- Weight matrices W_Q, W_K, W_V ∈ ℝ^(D×D) are partitioned into blocks W^(i,j)

### Advantages
- **Scalability**: Supports deployment on m × n devices (exceeds head-wise limits)
- **Load Balancing**: Even workload distribution across heads and dimensions
- **Memory Efficiency**: Each device stores only fraction of parameters/activations
- **Communication Efficiency**: Reduced cross-device synchronization

### Experimental Results
- **Setup**: 16 NVIDIA H100 GPUs, FP16 precision, batch size 1024
- **Models**: 2-layer Dense Transformer and 2-layer MoE Transformer (4 experts/layer)
- **Metrics**: Throughput (TPS) and Time Per Output Token (TPOT)

**Performance Gains:**
- Dense model: 31.7% throughput improvement (1.2M → 1.58M tokens/sec), 37.1% overhead reduction
- MoE model: 35.3% throughput improvement (850K → 1.15M tokens/sec), 33.3% overhead reduction

### Baseline Comparison
- Baseline: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2) on 16 GPUs
- Proposed: m×n=16 partitions (fully utilizing all 16 GPUs)
- Proposed method outperforms baseline in both throughput and communication efficiency