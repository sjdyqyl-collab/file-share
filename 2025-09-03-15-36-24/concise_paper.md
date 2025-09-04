# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction

Traditional MoE implementations colocate multiple experts per GPU to minimize communication, creating computational bottlenecks. Our approach shifts the optimization focus from reducing communication to maximizing compute concurrency by distributing experts across nodes with one expert per GPU.

## Methodology

### 1. Expert Placement Strategy
- **Single-expert-per-GPU**: Each GPU hosts at most one expert
- **Cross-node distribution**: Topology-aware placement considering bandwidth, latency, and memory
- **Large EP regime**: EP ≥ 16 for maximum parallelism

### 2. Routing and Load Balancing
- **Token batching**: Group tokens by destination expert
- **Asynchronous routing**: Overlap communication with computation
- **Dynamic load balancing**: Monitor and adjust gating probabilities

### 3. Communication Optimization
- **Compute-communication overlap**: Use CUDA streams and double buffering
- **Pipeline scheduling**: Micro-stages for each MoE layer
- **All-to-all communication**: Efficient token routing across nodes

## Experimental Setup

### Model Configuration
- **Architecture**: 4-layer MoE
- **Experts per layer**: 16 (baseline) vs 64 (proposed)
- **Expert type**: MLP with hidden size 32768
- **Precision**: FP16
- **Batch size**: 1024 tokens
- **MHA**: 16 heads × 512 dimensions

### Hardware
- **GPU**: NVIDIA H100 (80GB)
- **Network**: InfiniBand HDR (200 Gbps)
- **Baseline**: 16 GPUs (TP=8, PP=2)
- **Proposed**: 64 GPUs (EP=64)

## Results

| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts per GPU | 120,000 | 8.3ms |
| Proposed (Large EP) | 64 | 1 expert per GPU | 450,000 | 2.2ms |

**Improvements**: 3.75× higher throughput, 3.8× lower latency

## Key Advantages
1. **Maximized Expert Parallelism**: One expert per GPU eliminates contention
2. **Scalable Communication**: Overlap mitigates network overhead
3. **Near-linear Scaling**: 93.75% scaling efficiency from 16→64 GPUs
4. **Memory Efficiency**: Lower per-GPU memory usage (4.1GB vs 14.4GB)

## Conclusion

Large-scale cross-node expert parallelism with EP ≥ 16 achieves superior performance by maximizing compute concurrency. The method provides a scalable blueprint for high-performance MoE deployments in GPU-rich environments.