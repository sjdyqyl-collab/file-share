# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction
Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, scaling MoE models across large GPU clusters introduces challenges in expert placement and parallelization. Traditional strategies colocate multiple experts per GPU to reduce communication, creating computational bottlenecks that limit expert-level parallelism as model and cluster sizes grow.

We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This maximizes concurrent computation by allowing each expert to run in near isolation, shifting optimization from reducing communication to maximizing compute concurrency using modern HPC networking capabilities.

## Methods

### Expert Placement Strategy
- **Single-Expert-Per-GPU**: Deploy at most one expert per GPU, ensuring E ≤ G (experts ≤ GPUs)
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory capacity
- **Memory Optimization**: When E > G, replicate experts to maximize concurrency while balancing memory

### Routing and Load Balancing
- **Gating Mechanism**: Standard MoE top-K gating scores determine expert activation
- **Token Sharding**: Group tokens by destination expert, asynchronous routing, and dynamic load balancing
- **Load Monitoring**: Real-time per-expert load tracking with proportional gating probability adjustment

### Communication Overlap and Scheduling
- **Compute-Communication Overlap**: Interleave expert computation with asynchronous token transfers using CUDA streams/NCCL
- **Pipeline Scheduling**: Multi-layer MoE networks process tokens immediately upon arrival without waiting for full batches
- **Large EP Regime**: Optimized for EP ≥ 16 where network bandwidth becomes the limiting factor

### Scalability Considerations
- **Tensor Parallelism**: Optional 2-way TP within expert if memory constrained
- **Data Parallelism**: Applied across MoE network replicas for synchronized weight updates
- **Integration**: Compatible with large models exceeding single-GPU memory

## Experiments

### Setup
- **Model**: 4-layer MoE, 16 experts per layer (64 total), MLP experts, FP16 precision
- **Input**: 1024 sequences × 10000 tokens = 10.24M tokens per batch
- **Dimensions**: 16 attention heads × 512 dimensions, MLP hidden size 32768
- **Hardware**: H100 GPUs, inference-only setting

### Configurations
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed (Large EP) | 64 | 1 expert per GPU | 450,000 | 2.2ms |

### Results
- **3.75× higher throughput** (450K vs 120K tokens/second)
- **3.77× lower latency** (2.2ms vs 8.3ms per token)
- **93.75% scaling efficiency** despite 4× GPU increase
- **Near-linear scaling** achieved in large EP regime (EP ≥ 16)

## Conclusion
Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU, achieving significant performance improvements through asynchronous token routing and communication-computation overlap. The approach provides a scalable blueprint for high-performance MoE inference in GPU-rich environments, demonstrating 3.75× higher throughput with 64 H100 GPUs compared to traditional 16-GPU configurations.

## Deployment Configuration Summary
- **Baseline**: 16 H100 GPUs, TP=8, PP=2, 4 experts/GPU
- **Proposed**: 64 H100 GPUs, EP=64, 1 expert/GPU, topology-aware placement
- **Key Innovation**: Large EP (≥16) with cross-node expert distribution and asynchronous routing