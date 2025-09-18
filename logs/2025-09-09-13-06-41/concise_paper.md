# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## 1. Introduction
Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, scaling MoE models across large GPU clusters introduces challenges in expert placement and parallelization. Traditional approaches assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks that limit expert parallelism.

We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This design maximizes compute concurrency by leveraging modern HPC networking capabilities to sustain high bandwidth and low latency across nodes.

## 2. Methods

### 2.1 Expert Placement Strategy
- **Single-expert-per-GPU**: Each GPU hosts at most one expert
- **Cross-node distribution**: Experts distributed across nodes using topology-aware placement considering bandwidth, latency, and memory capacity
- **Mathematical constraint**: For E experts and G GPUs, ensure distinct GPU assignment when E ≤ G

### 2.2 Routing and Load Balancing
- **Gating mechanism**: Top-K expert selection (K=2) with softmax gating scores
- **Token sharding**: Group tokens by destination expert, asynchronous routing with dynamic load balancing
- **Load balancing**: Monitor per-expert load with exponential moving average adjustment when max/min ratio > 1.5

### 2.3 Communication Overlap and Scheduling
- **Compute-communication overlap**: CUDA streams separation with double buffering
- **Pipeline scheduling**: 4-layer MoE with micro-batches of 256 sequences
- **Asynchronous operations**: NCCL/MPI for non-blocking token transfers

### 2.4 Integration Strategies
- **Tensor parallelism**: Optional TP=2 when expert exceeds GPU memory
- **Pipeline parallelism**: 4 stages (one per layer) with token routing
- **Memory optimization**: FP16 precision, gradient checkpointing, memory pooling

## 3. Experiments

### 3.1 Setup
- **Model**: 4-layer MoE, 16 experts per layer, MLP experts
- **Precision**: FP16
- **Input**: 1024 sequences × 10,000 tokens = 10,240,000 tokens/batch
- **Dimensions**: Token=8192, MLP hidden=32768, 16 attention heads × 512
- **Hardware**: H100 GPUs with 80GB HBM3

### 3.2 Configurations
- **Baseline (TP=8, PP=2)**: 16 GPUs, 4 experts per GPU, colocated
- **Proposed**: 64 GPUs, 1 expert per GPU, cross-node distribution

### 3.3 Results
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert per GPU | 450,000 | 2.2ms |

**Improvements**: 3.75× higher throughput, 3.8× lower latency

### 3.4 Analysis
- **GPU utilization**: >90% with proposed vs 70% baseline
- **Communication**: Overlapped inter-node routing vs intra-GPU contention
- **Scalability**: Near-linear scaling at EP=64

## 4. Conclusion
Large-scale cross-node expert parallelism with one expert per GPU achieves significant performance improvements by maximizing expert-level parallelism and overlapping communication with computation. This approach provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.

## Key Technical Details
- **Large EP regime**: EP ≥ 16
- **Memory per expert**: 536MB parameters + 50GB activations
- **Network requirements**: 400 Gbps per GPU
- **Optimal configuration**: 64 GPUs for 64 experts (4 layers × 16 experts)