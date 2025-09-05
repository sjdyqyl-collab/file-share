# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction
Traditional MoE parallelization assigns multiple experts per GPU to reduce communication, creating computational bottlenecks that limit expert-level parallelism. Our large-scale cross-node expert parallelism method deploys at most one expert per GPU with EP ≥ 16, shifting the optimization focus from reducing communication to maximizing compute concurrency using modern HPC networking capabilities.

## Methodology

### Expert Placement Strategy
- **Single-Expert-Per-GPU**: Each GPU hosts at most one expert
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory
- **Allocation Rules**: 
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Experts replicated across GPUs while balancing memory

### Routing and Load Balancing
- **Gating**: Top-K gating scores determine expert activation per token
- **Token Sharding**: Group tokens by destination expert, asynchronous routing
- **Load Balancing**: Dynamic adjustment of gating probabilities to prevent overloading

### Communication Overlap
- **Asynchronous Operations**: CUDA streams and NCCL/MPI for overlapping compute and communication
- **Pipeline Scheduling**: Each MoE layer as micro-stage with fine-grained token routing
- **Topology-Aware Routing**: Minimize network congestion through intelligent placement

### Scalability Framework
- **Large EP Regime**: EP ≥ 16 with one-expert-per-GPU policy
- **Integration**: Compatible with TP (optional TP=2) and DP for memory-constrained scenarios
- **Memory Optimization**: Activation checkpointing and expert sharding

## Experiments

### Setup
- **Model**: 4-layer MoE, 16 experts/layer, 64 total experts
- **Architecture**: MHA (16 heads × 512 dim), MLP hidden size 32768
- **Precision**: FP16
- **Data**: 1024 sequences × 10,000 tokens = 10.24M tokens/batch
- **Hardware**: H100 GPUs, inference-only

### Configurations

#### Baseline (TP=8, PP=2)
- **GPUs**: 16 H100
- **Deployment**: 4 experts + TP shard per GPU
- **Parallelism**: 8-way TP, 2-stage PP
- **Expert Density**: 4 experts/GPU

#### Proposed (EP=64)
- **GPUs**: 64 H100
- **Deployment**: 1 expert per GPU
- **Parallelism**: 64-way EP, optional TP=2
- **Expert Density**: 1 expert/GPU

### Results
| Method | GPUs | TPS | TPOT | Improvement |
|--------|------|-----|------|-------------|
| Baseline | 16 | 120,000 | 8.3ms | - |
| Proposed | 64 | 450,000 | 2.2ms | 3.75× TPS, 3.8× latency |

## Conclusion
Large-scale cross-node expert parallelism with EP ≥ 16 achieves 3.75× higher throughput and 3.8× lower latency by dedicating one expert per GPU, overlapping communication with computation, and leveraging modern HPC networking for scalable MoE deployment.