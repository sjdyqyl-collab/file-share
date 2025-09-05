# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## 1. Introduction

Traditional MoE implementations suffer from computational bottlenecks due to multiple experts sharing GPU resources. Our approach shifts the optimization focus from communication reduction to compute concurrency maximization by distributing experts across nodes with one expert per GPU, achieving EP ≥ 16.

## 2. Methodology

### 2.1 Expert Placement Strategy
- **Single-expert-per-GPU**: Each GPU hosts exactly one expert
- **Cross-node distribution**: Topology-aware placement minimizing network congestion
- **Memory optimization**: 2.1GB per expert (FP16, hidden=32768)

### 2.2 Routing and Load Balancing
- **Dynamic gating**: Top-K expert selection (K=2)
- **Token batching**: Group tokens by destination expert (128-1024 tokens/batch)
- **Load balancing**: Monitor every 100 iterations, rebalance if imbalance > 1.5

### 2.3 Communication Overlap
- **CUDA streams**: Separate compute (stream 0) and communication (stream 1)
- **Pipeline scheduling**: 4-layer micro-pipeline with 128-token micro-batches
- **Asynchronous routing**: Overlap computation with cross-node token transfers

## 3. Experimental Setup

### 3.1 Model Configuration
- **Architecture**: 4-layer MoE, 16 experts/layer, MLP experts
- **Dimensions**: Hidden=32768, MHA: 16 heads × 512 dim/head
- **Precision**: FP16, Batch: 1024 sequences × 10K tokens

### 3.2 Deployments
- **Baseline**: 16 H100 GPUs, TP=8, PP=2, 4 experts/GPU
- **Proposed**: 64 H100 GPUs, EP=64, 1 expert/GPU

## 4. Results

| Method | GPUs | TPS | TPOT(ms) | Improvement |
|--------|------|-----|----------|-------------|
| Baseline | 16 | 120,000 | 8.3 | 1.0× |
| Proposed | 64 | 450,000 | 2.2 | 3.75× TPS, 3.8× latency |

**Key achievements**:
- 3.75× throughput improvement
- 3.8× latency reduction
- 95%+ GPU utilization vs 75% baseline
- Near-linear scaling in EP ≥ 16 regime

## 5. Deployment Configuration

### 5.1 Resource Requirements
- **GPUs**: 64× H100 (80GB each)
- **Network**: InfiniBand/NVSwitch (50+ GB/s)
- **Memory**: 13GB per GPU (expert + buffers)

### 5.2 Parallel Strategy
- **Primary**: Expert Parallelism (EP=64)
- **Optional**: Tensor Parallelism (TP=2 if needed)
- **Pipeline**: Layer-wise micro-stages

## 6. Conclusion

Large-scale cross-node expert parallelism with one-expert-per-GPU deployment achieves significant performance improvements by maximizing compute concurrency and minimizing expert contention, providing a scalable blueprint for high-performance MoE inference in GPU-rich environments.