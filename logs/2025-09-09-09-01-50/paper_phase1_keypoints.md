# Large-Scale Cross-Node Expert Parallelism for MoE Models - Key Points

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Technical Contributions

### 1. Expert Placement Strategy
- **Single-Expert-Per-GPU Deployment**: Deploy at most one expert per GPU to maximize expert-level parallelism
- **Cross-Node Distribution**: Topology-aware placement considering node-to-node bandwidth, GPU memory capacity, and token routing patterns
- **Large EP Regime**: EP ≥ 16 for maximum expert independence

### 2. Routing and Load Balancing
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities

### 3. Communication Overlap and Scheduling
- **Compute-Communication Overlap**: Interleave expert computation with cross-node token transfers
- **Pipeline Scheduling**: Fine-grained pipeline where experts start processing as soon as partial batches arrive
- **CUDA Streams/NCCL**: Leverage asynchronous communication to prevent blocking

## Experimental Setup
- **Model**: 4-layer MoE, 16 experts per layer, each expert is MLP
- **Precision**: FP16
- **Batch Size**: 1024 sequences
- **Sequence Length**: 10000 tokens
- **Token Dimension**: 8192
- **MHA**: 16 heads, 512 dimensions per head
- **MLP Hidden Size**: 32768

## Results Summary
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed Cross-Node EP | 64 | 1 expert per GPU | 450,000 | 2.2ms |

**Improvements**: 3.75× higher throughput, 3.8× lower latency

## Core Advantages
1. Maximized Expert Parallelism: One expert per GPU ensures minimal contention
2. Balanced Load: Topology-aware placement prevents bottlenecks
3. Scalable Communication: Asynchronous routing enables near-linear scaling for EP ≥ 16
4. Model Compatibility: Integrates with TP and DP for large models