# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models: Concise Version

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction
Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, traditional MoE parallelization strategies assign multiple experts to the same GPU, creating computational bottlenecks that limit expert-level parallelism as model and cluster sizes grow. We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond to unlock higher degrees of concurrent computation.

## Methods

### Expert Placement Strategy
- **Single-Expert-Per-GPU**: Deploy at most one expert per GPU to eliminate intra-GPU contention
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory
- **Large EP Regime**: EP ≥ 16 with experts distributed across as many devices as possible

### Routing and Load Balancing
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Non-blocking token transfers with dynamic gating adjustment
- **Load Monitoring**: Real-time per-expert load balancing to prevent overloading

### Communication Overlap
- **Compute-Communication Overlap**: Use CUDA streams and NCCL/MPI for asynchronous operations
- **Pipeline Scheduling**: Fine-grained micro-stages with immediate token routing between layers
- **Topology Optimization**: Minimize maximum tokens sent across any single network link

## Experiments

### Setup
- **Model**: 4-layer MoE, 16 experts per layer, MLP experts with hidden size 32,768
- **Precision**: FP16, batch size 1024 tokens
- **MHA**: 16 heads × 512 dimensions per head
- **Hardware**: H100 GPUs with high-bandwidth interconnects

### Configurations
| Method | GPUs | Parallel Strategy | Per-GPU Deployment | TPS | TPOT (ms) |
|--------|------|------------------|-------------------|-----|-----------|
| Baseline | 16 | TP=8, PP=2 | 4 experts + TP shard | 120,000 | 8.3 |
| Proposed | 64 | EP=64 | 1 expert per GPU | 450,000 | 2.2 |

### Results
- **3.75× throughput increase** (450,000 vs 120,000 TPS)
- **3.8× latency reduction** (2.2ms vs 8.3ms TPOT)
- **Near-linear scaling** with 64 GPUs in large EP regime
- **Full GPU utilization** through dedicated expert allocation

## Conclusion
Our large-scale cross-node expert parallelism method achieves significant performance improvements by maximizing expert-level parallelism through one-expert-per-GPU deployment. The approach successfully shifts the bottleneck from intra-GPU contention to manageable communication overhead, validated through 3.75× throughput gains and 3.8× latency reduction in HPC environments with EP ≥ 16.

## Key Technical Specifications
- **Expert Architecture**: MLP with 32,768 hidden dimensions
- **Parallelism**: EP=64 (64 experts across 64 GPUs)
- **Communication**: NCCL/MPI with CUDA streams for overlap
- **Load Balancing**: Dynamic gating with real-time adjustment
- **Scalability**: Linear scaling demonstrated with H100 clusters