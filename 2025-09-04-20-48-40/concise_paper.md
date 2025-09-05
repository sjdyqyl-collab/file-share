# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction
Traditional MoE implementations colocate multiple experts on the same GPU to reduce communication overhead, creating computational bottlenecks that limit expert-level parallelism. As model and cluster sizes grow, this trade-off becomes increasingly suboptimal. We present a cross-node expert parallelism method that prioritizes distributing experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This shifts the optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities.

## Methodology

### Expert Placement Strategy
- **Single-Expert-Per-GPU**: Each GPU hosts at most one expert, ensuring E ≤ G where E=experts and G=GPUs
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory capacity
- **Large EP Regime**: EP ≥ 16 for maximum parallelism

### Routing and Load Balancing
- **Asynchronous Token Routing**: Tokens batched by destination expert with async communication
- **Dynamic Load Balancing**: Real-time monitoring and gating probability adjustments
- **Communication Overlap**: CUDA streams/NCCL for overlapping compute and communication

### Pipeline Scheduling
- **Micro-stage Pipeline**: Each MoE layer as separate stage
- **Fine-grained Overlap**: Partial batch processing without waiting for full completion
- **Immediate Routing**: Token outputs routed directly to next layer's experts

## Experiments

### Setup
- **Model**: 4-layer MoE, 16 experts/layer, MLP experts
- **Precision**: FP16, Batch: 1024 tokens
- **Dimensions**: 16 heads × 512 = 8192 attention, 32768 MLP hidden
- **Hardware**: H100 GPUs

### Configurations
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts/GPU + TP shard | 120K | 8.3ms |
| Proposed (EP=64) | 64 | 1 expert/GPU | 450K | 2.2ms |

### Results
- **3.75× throughput improvement** (450K vs 120K tokens/sec)
- **3.8× latency reduction** (2.2ms vs 8.3ms per token)
- **Linear scalability** achieved with EP=64 ≥ 16
- **Full GPU utilization** without expert contention

## Conclusion
Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU, achieving significant performance improvements through asynchronous token routing and communication-computation overlap. The approach provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.

## Technical Specifications
- **Parallelism**: EP=64, optional TP=2, micro-stage PP=4
- **Communication**: NCCL async, CUDA streams, topology-aware routing
- **Load Balancing**: Dynamic gating with 100-step update frequency
- **Memory**: FP16 precision, single-GPU expert placement
- **Network**: 400Gbps InfiniBand, 1μs latency, NVLink intra-node