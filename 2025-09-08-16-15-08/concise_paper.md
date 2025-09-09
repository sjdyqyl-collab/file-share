# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## 1. Introduction

Traditional MoE parallelization assigns multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks that limit expert-level parallelism. We present a cross-node expert parallelism method that distributes experts such that each GPU hosts at most one expert, pushing EP to 16 or beyond to unlock higher degrees of concurrent computation.

## 2. Methods

### 2.1 Expert Placement Strategy
- **Single-Expert-Per-GPU**: Deploy at most one expert per GPU to eliminate intra-GPU contention
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory capacity
- **Large EP Regime**: EP ≥ 16 for maximum expert independence

### 2.2 Routing and Load Balancing
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Overlap communication with computation using NCCL/MPI
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities

### 2.3 Communication Overlap
- **Compute-Communication Interleaving**: Process tokens while transferring next batch
- **Pipeline Scheduling**: Each MoE layer as micro-stage with immediate token routing
- **Fine-grained Overlap**: Minimize GPU idle time across layers

## 3. Experiments

### 3.1 Model Configuration
- **Architecture**: 4-layer MoE, 16 experts per layer (MLP)
- **Precision**: FP16
- **Dimensions**: Token dim=8192, MLP hidden=32768, MHA=16×512 heads
- **Batch**: 1024 sequences × 10,000 tokens

### 3.2 Deployment Comparison

| Method | GPUs | Configuration | TPS | TPOT |
|--------|------|---------------|-----|------|
| Baseline | 16 | TP=8, PP=2, 4 experts/GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert/GPU, EP=16 | 450,000 | 2.2ms |

### 3.3 Results
- **3.75× throughput improvement** (450k vs 120k TPS)
- **3.8× latency reduction** (2.2ms vs 8.3ms TPOT)
- **Near-linear scaling** with 64 GPUs in large EP regime

## 4. Conclusion

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU, achieving significant throughput and latency improvements. The approach provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.