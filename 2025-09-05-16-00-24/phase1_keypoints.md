# Phase 1: Key Points Extraction

## Original Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Points

### Core Problem
- Traditional MoE implementations colocate multiple experts on the same GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- The trade-off becomes suboptimal as model and cluster sizes grow

### Proposed Solution
- **Large-scale cross-node expert parallelism** for MoE models
- **One expert per GPU** deployment strategy
- **Large EP regime**: EP ≥ 16 experts per parallel group
- Focus shifts from reducing communication to maximizing compute concurrency

### Technical Innovations
1. **Expert Placement Strategy**: One expert per GPU, distributed across nodes
2. **Routing and Load Balancing**: Dynamic token routing with asynchronous batching
3. **Communication Overlap**: Interleaving computation and communication using CUDA streams
4. **Topology-aware placement** considering bandwidth, latency, and memory capacity

### Model Architecture
- 4-layer Mixture-of-Experts (MoE)
- 16 experts per layer
- Each expert is a MLP
- FP16 precision
- Hidden size of MLP: 32768
- MHA: 16 heads, 512 dimensions per head

### Experimental Configuration
- **Baseline**: 16 H100 GPUs, TP=8, PP=2, 4 experts per GPU
- **Proposed**: 64 H100 GPUs, 1 expert per GPU, EP=64
- Batch size: 1024 sequences
- Sequence length: 10000 tokens per sequence

### Performance Results
- **Throughput**: 450,000 TPS (vs 120,000 TPS baseline) - 3.75× improvement
- **Latency**: 2.2ms TPOT (vs 8.3ms baseline) - 3.8× improvement
- **Scalability**: Near-linear scaling in large EP regime (EP ≥ 16)

### Deployment Strategy
- **Parallelism**: Expert Parallelism (EP) as primary dimension
- **Optional**: Tensor Parallelism (TP=2) if expert doesn't fit on single GPU
- **Pipeline**: Each MoE layer as micro-stage with overlapped communication
- **Communication**: Asynchronous token routing with batching

### Advantages
1. Maximized Expert Parallelism: One expert per GPU ensures minimal contention
2. Balanced Load: Topology-aware placement prevents network bottlenecks
3. Scalable Communication: Asynchronous routing enables near-linear scaling
4. Large Model Support: Integrates with TP and DP for memory constraints

### Target Environment
- High-performance computing (HPC) environments
- Large GPU clusters (H100-class)
- Environments with abundant GPU resources
- Advanced networking (NVLink, InfiniBand, NVSwitch)