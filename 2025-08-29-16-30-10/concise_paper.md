# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, scaling MoE models across large GPU clusters introduces challenges in expert placement and parallelization. Traditional strategies colocate multiple experts on the same GPU to reduce inter-node communication, creating computational bottlenecks that limit expert-level parallelism as model and cluster sizes grow.

We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This design shifts the optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities.

## Methods

### Overview
Our approach maximizes expert-level parallelism through three key components: (1) Expert Placement Strategy, (2) Routing and Load Balancing, and (3) Communication Overlap and Scheduling.

### Expert Placement Strategy

**Single-Expert-Per-GPU Deployment**: Deploy at most one expert per GPU. For E experts and G GPUs, assign each expert to a distinct GPU if E ≤ G. If E > G, replicate experts to maximize concurrency while balancing memory usage.

**Cross-Node Distribution**: Use topology-aware placement considering node-to-node bandwidth, latency, GPU memory capacity, and expected token routing patterns to minimize hotspotting.

### Routing and Load Balancing

**Gating Mechanism**: Standard top-K gating scores determine expert activation, with dynamic adjustment of gating probabilities to prevent overloading specific experts.

**Token Sharding**: Group tokens by destination expert, use asynchronous routing to overlap computation, and ensure balanced workload across all experts.

### Communication Overlap and Scheduling

**Overlapping Compute and Communication**: Interleave expert computation with token transfers using CUDA streams or asynchronous communication libraries (NCCL/MPI).

**Pipeline Scheduling**: In multi-layer MoE networks, immediately route token outputs to next layer's experts, starting processing on partial batches to reduce idle time.

### Scalability Considerations

**Large EP Regime (EP ≥ 16)**: Optimized for configurations with 16+ experts per parallel group, where network bandwidth becomes the primary limiting factor.

**Memory Integration**: Support tensor model parallelism within individual experts if they exceed GPU memory, combined with data parallelism across MoE network replicas.

## Experiments

### Setup
- **Model**: 4-layer MoE, 16 experts per layer (MLP experts)
- **Precision**: FP16
- **Batch Size**: 1024 tokens per forward pass
- **Dimensions**: 16 attention heads × 512 dimensions, MLP hidden size 32768
- **Hardware**: H100 GPUs, inference-only

### Deployment Configurations

**Baseline (TP=8, PP=2)**:
- 16 H100 GPUs total
- Tensor parallelism: 8-way sharding across GPUs
- Pipeline parallelism: 2 stages (8 GPUs each)
- Expert colocation: 4 experts per GPU
- Sequential token processing through pipeline stages

**Proposed Cross-Node Expert Parallelism**:
- 64 H100 GPUs total
- Expert parallelism: 64 (one GPU per expert per layer)
- Each GPU hosts exactly one expert
- Optional tensor parallelism (TP=2) if expert exceeds memory
- Dynamic token routing with asynchronous communication
- All 64 experts per layer compute in parallel

### Results

| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert per GPU | 450,000 | 2.2ms |

**Performance Gains**: 3.75× higher throughput and 3.8× lower latency than baseline.

## Conclusion

We proposed a large-scale cross-node expert parallelism method that maximizes expert-level parallelism by deploying at most one expert per GPU. Our approach achieved 3.75× higher throughput and 3.8× lower latency by fully utilizing 64 GPUs for a 4-layer, 64-expert-per-layer MoE model. This provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.