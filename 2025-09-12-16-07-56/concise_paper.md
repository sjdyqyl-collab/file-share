# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction
Traditional MoE parallelization assigns multiple experts to the same GPU to reduce communication, creating computational bottlenecks. Network advances (NVLink, InfiniBand) make communication less dominant, shifting focus to maximizing compute concurrency through large expert parallelism (EP ≥ 16).

## Methods

### Expert Placement Strategy
- **Constraint**: At most one expert per GPU
- **Rule**: If E ≤ G, each expert to distinct GPU; if E > G, replicate with memory balancing
- **Topology-aware**: Considers bandwidth, latency, memory capacity, routing patterns

### Routing and Load Balancing
- **Gating**: Top-2 expert selection per token
- **Token sharding**: Group by destination expert, asynchronous routing
- **Dynamic balancing**: Monitor load, adjust gating probabilities (threshold=1.5)

### Communication Overlap
- **CUDA streams**: Separate compute and communication streams
- **Pipeline scheduling**: Immediate routing between layers
- **Asynchronous**: Overlap computation with token transfers

### Memory Integration
- **Tensor parallelism**: TP=2 fallback if expert exceeds memory
- **Data parallelism**: Standard DP across replicas

## Experiments

### Setup
- **Model**: 4-layer MoE, 16 experts/layer, 64 total experts
- **Precision**: FP16
- **Batch**: 1024 sequences × 10000 tokens = 10.24M tokens
- **Dimensions**: Token=8192, MHA=16×512, MLP hidden=32768
- **Hardware**: H100 GPUs

### Configurations
| Method | GPUs | Strategy | TPS | TPOT |
|--------|------|----------|-----|------|
| Baseline | 16 | TP=8, PP=2, 4 experts/GPU | 120,000 | 8.3ms |
| Proposed | 64 | EP=64, 1 expert/GPU | 450,000 | 2.2ms |

### Results
- **3.75× throughput improvement**
- **3.8× latency reduction**
- **93.75% scaling efficiency** (3.75× on 4× GPUs)

## Conclusion
Large-scale cross-node expert parallelism with one expert per GPU achieves significant performance gains by maximizing compute concurrency while effectively managing communication overhead through asynchronous routing and overlap techniques.

## Key Parameters
- **Expert count**: 16 per layer × 4 layers = 64
- **Token dimension**: 8192
- **MLP hidden**: 32768
- **Batch size**: 1024 sequences × 10000 tokens
- **Precision**: FP16
- **Minimum EP**: 16
- **Network**: InfiniBand ≥50 GB/s, NVLink ≥300 GB/s
- **GPU memory**: ~2-4 GB per expert

## Critical Dimensions
- **Model**: 4-layer MoE, 64 experts total
- **Input**: 10.24M tokens/batch (1024×10000)
- **Compute**: 64 H100 GPUs (1 expert/GPU)
- **Performance**: 450K TPS, 2.2ms TPOT