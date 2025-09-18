# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction
Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, traditional MoE parallelization strategies assign multiple experts to the same GPU, creating computational bottlenecks that limit expert-level parallelism. We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This design maximizes compute concurrency by leveraging modern HPC networking capabilities to sustain high bandwidth and low latency across nodes.

## Background
MoE models replace transformer FFN layers with multiple "experts," each specializing in different input patterns. Standard implementations use moderate EP degrees with multiple experts per GPU to limit communication. However, as network interconnects advance, the communication cost becomes less dominant than gains from maximizing compute concurrency. We define *large EP* as configurations where EP ≥ 16, distributing experts across as many devices as possible—ideally one per GPU—to minimize resource contention and maximize parallel execution.

## Methods

### Expert Placement Strategy
**Single-Expert-Per-GPU Deployment**: Deploy at most one expert per GPU. For E experts and G GPUs, assign each expert to a distinct GPU if E ≤ G. If E > G, replicate experts to maximize concurrency while balancing memory usage.

**Cross-Node Distribution**: Use topology-aware placement considering node-to-node bandwidth, GPU memory capacity, and expected token routing patterns to minimize network link utilization.

### Routing and Load Balancing
**Token Sharding**: Group tokens by destination expert to reduce network messages. Send token batches asynchronously to overlap with expert computation. Monitor per-expert load and dynamically adjust gating probabilities to prevent overloading.

### Communication Overlap and Scheduling
**Compute-Communication Overlap**: Interleave expert computation with communication using CUDA streams or asynchronous libraries (NCCL/MPI). While one batch processes, the next batch transfers simultaneously.

**Pipeline Scheduling**: Route token outputs immediately to next layer's experts, starting processing on partial batches rather than waiting for full batches.

### Scalability Considerations
**Large EP Regime (EP ≥ 16)**: Network bandwidth becomes the primary limiting factor, mitigated through topology-aware routing and token batching. One-expert-per-GPU ensures full GPU utilization while amortizing communication costs.

**Integration**: Compatible with tensor parallelism (TP) within single experts and data parallelism (DP) across MoE replicas.

## Experiments

### Setup
- **Model**: 4-layer MoE, 16 experts/layer, MLP experts
- **Precision**: FP16
- **Batch**: 1024 sequences × 10,000 tokens = 10.24M tokens/batch
- **Dimensions**: Token dim=8192, MHA=16×512=8192, MLP hidden=32768
- **Hardware**: H100 GPUs (inference-only)

### Deployment Configurations

**Baseline (TP=8, PP=2)**:
- 16 H100 GPUs
- Each GPU: 1/8 tensor-parallel shard + 4 colocated experts
- 2 pipeline stages, 8 GPUs each

**Proposed Cross-Node Expert Parallelism**:
- 64 H100 GPUs
- Each GPU: exactly one expert
- EP=64, optional TP=2 within expert
- Asynchronous token routing with overlapped communication

### Results
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline | 16 | 4 experts + TP shard/GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert/GPU | 450,000 | 2.2ms |

**Performance**: 3.75× higher throughput, 3.8× lower latency with 4× GPU usage achieving near-linear scaling.

## Conclusion
Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU. Through topology-aware placement, asynchronous routing, and communication-computation overlap, we achieve 3.75× throughput improvement and 3.8× latency reduction in the large EP regime. This provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.