# Phase 1: Key Points Extraction

## Abstract (Retained as-is)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Technical Contributions

### 1. Expert Placement Strategy
- **Single-expert-per-GPU deployment**: Each GPU hosts at most one expert
- **Cross-node distribution**: Experts distributed across nodes to minimize hotspotting
- **Topology-aware placement**: Considers node-to-node bandwidth, latency, GPU memory capacity, and token routing patterns

### 2. Routing and Load Balancing
- **Gating mechanism**: Top-K gating scores determine expert activation
- **Token sharding**: Group tokens by destination expert, asynchronous routing, dynamic load balancing
- **Load balancing**: Monitor per-expert load and adjust gating probabilities dynamically

### 3. Communication Optimization
- **Overlapping compute and communication**: Interleave expert computation with token transfers
- **Pipeline scheduling**: Fine-grained pipeline for multi-layer MoE networks
- **Asynchronous communication**: CUDA streams/NCCL/MPI for non-blocking transfers

## Experimental Setup (Key Parameters)
- **Model**: 4-layer MoE, 16 experts per layer, each expert is MLP
- **Precision**: FP16
- **Batch size**: 1024 sequences
- **Sequence Length**: 10000 tokens per sequence
- **Token Dimension**: 8192
- **MHA**: 16 heads, 512 dimension per head
- **MLP Hidden Size**: 32768

## Results Summary
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert per GPU | 450,000 | 2.2ms |

## Key Performance Metrics
- **3.75× higher throughput** than baseline
- **3.8× lower latency** than baseline
- **Large EP regime**: EP ≥ 16
- **Scalability**: Near-linear scaling with 64 GPUs