# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models - Concise Version

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction
Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per token. However, scaling MoE models across GPU clusters introduces challenges in expert placement and parallelization. Traditional approaches colocate multiple experts on the same GPU to reduce communication, but this creates computational bottlenecks and limits expert parallelism.

This work presents a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, pushing EP to 16 or beyond. This maximizes concurrent computation while leveraging modern HPC networking to handle communication overhead.

## Methods

### Expert Placement Strategy
- **Single-expert-per-GPU**: Deploy at most one expert per GPU
- **Cross-node distribution**: Topology-aware placement considering bandwidth, latency, and memory
- **Replication**: When E > G, replicate experts to maximize concurrency

### Routing and Load Balancing
- **Token batching**: Group tokens by destination expert
- **Asynchronous routing**: Non-blocking communication overlapping computation
- **Dynamic load balancing**: Adjust gating probabilities based on expert load

### Communication Overlap
- **CUDA streams**: Separate computation and communication streams
- **Pipeline scheduling**: Fine-grained scheduling across MoE layers
- **Double buffering**: Overlap current computation with next batch transfer

### Scalability Considerations
- **Large EP regime**: EP ≥ 16 for maximum parallelism
- **Integration**: Compatible with TP and DP for large models
- **Memory management**: Efficient GPU memory usage per expert

## Experiments

### Setup
- **Model**: 4-layer MoE, 16 experts per layer, MLP experts
- **Precision**: FP16
- **Batch**: 1024 sequences × 10,000 tokens
- **Dimensions**: 16 heads × 512 dim, 32,768 MLP hidden
- **Hardware**: H100 GPUs

### Results
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert per GPU | 450,000 | 2.2ms |

**Performance**: 3.75× higher throughput, 3.8× lower latency

## Conclusion
The proposed method maximizes expert-level parallelism by deploying one expert per GPU, achieving significant performance improvements in large-scale MoE deployments. The approach scales effectively in the large EP regime (EP ≥ 16) and provides a blueprint for future high-performance MoE systems.

## Critical Parameters
- EP degree: ≥ 16
- Experts per layer: 64 (experimental)
- Hidden dimension: 32,768
- Batch size: 1024 × 10,000 tokens
- Precision: FP16
- Hardware: H100 cluster with NVLink/InfiniBand