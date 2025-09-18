# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models - Key Points

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Points

### Problem Statement
- Traditional MoE parallelization assigns multiple experts per GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- Need to shift from communication optimization to compute concurrency maximization

### Proposed Solution
- **Large Expert Parallelism (EP ≥ 16)**: Deploy at most one expert per GPU
- **Cross-node distribution**: Distribute experts across nodes to exploit all compute resources
- **Communication-computation overlap**: Use asynchronous routing and pipeline scheduling

### Core Components
1. **Expert Placement Strategy**: One expert per GPU, topology-aware placement
2. **Routing and Load Balancing**: Dynamic gating with token batching and asynchronous routing
3. **Communication Overlap**: CUDA streams/NCCL for overlapping compute and communication

### Technical Details
- **Model**: 4-layer MoE, 16 experts per layer, each expert is MLP
- **Precision**: FP16
- **Batch**: 1024 sequences × 10000 tokens = 10.24M tokens
- **Dimensions**: Token dim = 8192, MHA heads = 16, head dim = 512, MLP hidden = 32768

### Deployment Configurations
- **Baseline**: TP=8, PP=2, 16 H100 GPUs, 4 experts per GPU
- **Proposed**: EP=64, 64 H100 GPUs, 1 expert per GPU

### Results
- **Throughput**: 450,000 TPS (vs 120,000 TPS baseline) - 3.75× improvement
- **Latency**: 2.2ms TPOT (vs 8.3ms baseline) - 3.8× improvement
- **Scalability**: Near-linear scaling in large EP regime (EP ≥ 16)

### Advantages
1. Maximized expert parallelism (one expert per GPU)
2. Balanced load across nodes
3. Scalable communication overlap
4. Compatible with TP/DP for memory constraints