# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction
Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, traditional MoE parallelization strategies assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks that limit expert-level parallelism. We present a cross-node expert parallelism method that prioritizes distributing experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond to unlock higher degrees of concurrent computation.

## Methodology

### Expert Placement Strategy
**Single-Expert-Per-GPU Deployment**: Deploy at most one expert per GPU. For E experts and G GPUs:
- If E ≤ G: Each expert assigned to distinct GPU
- If E > G: Replicate experts across GPUs to maximize concurrency

**Cross-Node Distribution**: Topology-aware placement considering node-to-node bandwidth, latency, GPU memory capacity, and expected token routing patterns to minimize maximum tokens sent across any single link.

### Routing and Load Balancing
**Token Sharding**: Group tokens by destination expert to reduce network messages. Asynchronously send token batches to overlap with computation. Monitor per-expert load and dynamically adjust gating probabilities to prevent overloading.

### Communication Overlap and Scheduling
**Overlapping Compute and Communication**: While one batch processes on GPU, next batch transfers from other nodes using CUDA streams or NCCL/MPI. **Pipeline Scheduling**: Each MoE layer acts as a micro-stage; experts start processing partial batches immediately upon arrival.

### Scalability Considerations
**Large EP Regime**: Optimized for EP ≥ 16 where network bandwidth is primary limiting factor. Integrates with tensor model parallelism (TP) within experts and data parallelism (DP) across replicas.

## Experiments

### Setup
- **Model**: 4-layer MoE, 16 experts per layer (MLP), FP16
- **Input**: 1024 sequences × 10000 tokens = 10.24M tokens/batch
- **Dimensions**: Token dim=8192, MLP hidden=32768, 16 heads × 512 dim/head
- **Hardware**: H100 GPUs with high-bandwidth interconnects

### Configurations
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed Cross-Node EP | 64 | 1 expert per GPU | 450,000 | 2.2ms |

### Results
- **3.75× higher throughput** (450k vs 120k TPS)
- **3.8× lower latency** (2.2ms vs 8.3ms TPOT)
- **Near-linear scaling** with 4× more GPUs

## Conclusion
Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU. By shifting the bottleneck from intra-GPU contention to communication (effectively mitigated through asynchronous routing and overlap), we achieve 3.75× higher throughput and 3.8× lower latency in a 64-GPU setup, providing a scalable blueprint for high-performance MoE inference.