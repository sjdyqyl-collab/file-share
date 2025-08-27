# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, traditional MoE parallelization strategies assign multiple experts to the same GPU, creating computational bottlenecks and limiting expert-level parallelism as cluster sizes grow. We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond to unlock higher degrees of concurrent computation.

## Methods

### Expert Placement Strategy
- **Single-Expert-Per-GPU**: Deploy at most one expert per GPU to eliminate intra-GPU contention
- **Cross-Node Distribution**: Use topology-aware placement considering bandwidth, latency, GPU memory, and routing patterns
- **Expert Assignment**: For E experts and G GPUs, assign each expert to distinct GPU if E ≤ G; replicate experts across GPUs when E > G while maximizing concurrency

### Routing and Load Balancing
- **Gating Mechanism**: Standard top-K gating determines expert activation per token
- **Token Sharding**: Group tokens by destination expert, use asynchronous routing, and dynamically adjust gating probabilities for load balancing
- **Load Monitoring**: Continuously monitor per-expert load to prevent stragglers

### Communication Overlap and Scheduling
- **Compute-Communication Overlap**: Use CUDA streams or NCCL/MPI to interleave expert computation with cross-node token transfers
- **Pipeline Scheduling**: Process partial batches immediately in subsequent layers without waiting for full batch completion
- **Asynchronous Operations**: Ensure data transfer doesn't block GPU computation

### Scalability Considerations
- **Large EP Regime**: Optimized for EP ≥ 16 where network bandwidth is the primary limiting factor
- **Memory Integration**: Apply tensor parallelism within GPU only if expert exceeds memory capacity; use data parallelism across MoE replicas
- **Network Requirements**: Leverage NVLink, InfiniBand, and NVSwitch fabrics for high bandwidth and low latency

## Experiments

### Setup
- **Model**: 4-layer MoE with 16 experts per layer (MLP experts)
- **Precision**: FP16
- **Batch Size**: 1024 tokens per forward pass
- **Hardware**: H100 GPUs
- **Metrics**: TPS (Tokens per Second), TPOT (Time per Output Token)

### Configurations

| Method | GPUs | Expert Placement | TPS | TPOT |
|--------|------|------------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts per GPU | 120,000 | 8.3 ms |
| Proposed (EP=64) | 64 | 1 expert per GPU | 450,000 | 2.2 ms |

### Results
- **Throughput**: 3.75× improvement (450k vs 120k TPS)
- **Latency**: 3.8× reduction (2.2ms vs 8.3ms TPOT)
- **Scalability**: Near-linear scaling achieved with 64 GPUs in large EP regime

## Conclusion

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU across nodes. This approach shifts the computational bottleneck from intra-GPU contention to communication, effectively mitigated through asynchronous token routing and computation-communication overlap. The method achieves 3.75× higher throughput and 3.8× lower latency compared to traditional approaches, providing a scalable blueprint for high-performance MoE deployments in GPU-rich environments.