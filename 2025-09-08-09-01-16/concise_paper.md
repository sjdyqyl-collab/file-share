# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models: Concise Version

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Problem Statement

Traditional MoE parallelization assigns multiple experts per GPU to reduce communication, creating computational bottlenecks and limiting expert-level parallelism as model/cluster sizes grow.

## Proposed Solution

Large-scale cross-node expert parallelism with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond.

## Methodology

### Expert Placement Strategy
- **Single-Expert-Per-GPU**: At most one expert per GPU
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory
- **Condition**: If E ≤ G, each expert assigned to distinct GPU; if E > G, experts replicated for maximum concurrency

### Routing and Load Balancing
- **Gating**: Top-K dynamic gating with load balancing
- **Token Sharding**: Group tokens by destination expert, asynchronous routing, dynamic load adjustment
- **Communication**: Asynchronous cross-node token transfer with computation overlap

### Communication Overlap and Scheduling
- **Compute-Communication Interleaving**: Process current batch while transferring next batch
- **Pipeline Scheduling**: Fine-grained processing with partial batch arrival
- **Technology**: CUDA streams, NCCL/MPI for asynchronous communication

### Scalability Features
- **Large EP Regime**: EP ≥ 16 with network bandwidth as primary limiter
- **Memory Integration**: Optional TP=2 within expert if needed, DP across replicas
- **Topology Awareness**: Bandwidth and latency optimized placement

## Experimental Setup

### Model Configuration
- **Architecture**: 4-layer MoE with 16 experts per layer (64 total experts)
- **Expert Type**: MLP
- **Precision**: FP16
- **Batch**: 1024 sequences
- **Sequence**: 10,000 tokens
- **Dimensions**: 8192 token, 16 MHA heads (512 per head), 32,768 MLP hidden

### Hardware
- **GPUs**: H100
- **Setting**: Inference-only

## Deployment Configurations

### Baseline (TP=8, PP=2)
- **GPUs**: 16
- **Parallelism**: TP=8, PP=2
- **Per-GPU**: 4 experts + 1/8 tensor shard
- **Pipeline**: 2 stages (8 GPUs each), layers [0,1] and [2,3]

### Proposed (Large EP)
- **GPUs**: 64
- **Parallelism**: EP=64 (one expert per GPU)
- **Layer-Expert Mapping**: Each layer's 16 experts on 16 distinct GPUs
  - Layer 0: GPUs 0-15
  - Layer 1: GPUs 16-31
  - Layer 2: GPUs 32-47
  - Layer 3: GPUs 48-63
- **Features**: Asynchronous routing, topology-aware placement, communication overlap

## Results

| Method | GPUs | TPS | TPOT (ms) | Improvement |
|--------|------|-----|-----------|-------------|
| Baseline | 16 | 120,000 | 8.3 | - |
| Proposed | 64 | 450,000 | 2.2 | 3.75× throughput, 3.8× latency reduction |

## Key Advantages

1. **Maximized Expert Parallelism**: One expert per GPU eliminates contention
2. **Scalable Communication**: Asynchronous routing with computation overlap
3. **Near-Linear Scaling**: 4× GPUs yielding 3.75× throughput
4. **Large EP Regime**: EP=64 demonstrates effectiveness of large expert parallelism
5. **Topology Awareness**: Optimized for HPC environments with high-bandwidth interconnects

## Conclusion

Our large-scale cross-node expert parallelism method achieves significant performance improvements by maximizing expert-level parallelism through one-expert-per-GPU deployment. The approach successfully shifts the bottleneck from intra-GPU contention to communication, effectively mitigated through asynchronous routing and computation overlap. This provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.