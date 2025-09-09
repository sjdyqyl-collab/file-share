# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction
Mixture-of-Experts architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, traditional MoE parallelization strategies assign multiple experts to the same GPU, creating computational bottlenecks and limiting expert-level parallelism. We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, pushing EP to 16 or beyond to unlock higher degrees of concurrent computation.

## Methods

### Expert Placement Strategy
- **Single-Expert-Per-GPU**: Each GPU hosts at most one expert, ensuring no intra-GPU contention
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, memory capacity, and routing patterns
- **Large EP Regime**: EP ≥ 16 for maximum expert independence

### Routing and Load Balancing
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with computation
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to maintain <5% imbalance

### Communication Overlap and Scheduling
- **Compute-Communication Overlap**: Interleave expert computation with cross-node token transfers using CUDA streams
- **Pipeline Scheduling**: Fine-grained pipeline where experts start processing partial batches immediately
- **Topology-Aware Routing**: Minimize network hops and maximize bandwidth utilization

### Scalability Framework
- **Network Requirements**: 50 GB/s bandwidth, <15μs latency
- **Memory Integration**: Optional TP=2 within expert if FFN exceeds GPU memory
- **Combined Parallelism**: DP × EP × TP × PP for large models

## Experiments

### Setup
- **Model**: 4-layer MoE, 16 experts/layer, MLP experts
- **Precision**: FP16
- **Input**: 1024 sequences × 10000 tokens × 8192 dimensions
- **Hardware**: H100 GPUs, InfiniBand HDR

### Configurations
| Method | GPUs | Parallelism | Expert/GPU | TPS | TPOT |
|--------|------|-------------|------------|-----|------|
| Baseline | 16 | TP=8, PP=2 | 4 experts | 120,000 | 8.3ms |
| Proposed | 64 | EP=64 | 1 expert | 450,000 | 2.2ms |

### Results
- **3.75× throughput improvement** (450K vs 120K tokens/sec)
- **3.8× latency reduction** (2.2ms vs 8.3ms per token)
- **94% scaling efficiency** from 16 to 64 GPUs
- **95% GPU utilization** with no expert contention

## Conclusion
Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU, achieving significant throughput and latency improvements. The approach provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.