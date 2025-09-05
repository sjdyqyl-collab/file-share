# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction
Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, traditional MoE parallelization strategies assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks and limiting expert-level parallelism. We present a cross-node expert parallelism method that prioritizes distributing experts across nodes with at most one expert per GPU, pushing EP to 16 or beyond to unlock higher degrees of concurrent computation.

## Methods

### Expert Placement Strategy
- **Single-expert-per-GPU**: Each GPU hosts at most one expert
- **Cross-node distribution**: Experts distributed across nodes using topology-aware placement
- **Assignment rule**: If E ≤ G, each expert gets distinct GPU; if E > G, replicate experts while balancing memory

### Routing and Load Balancing
- **Top-K routing**: K=2 gating scores determine expert activation
- **Token batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous routing**: Overlap token transfer with expert computation
- **Dynamic load balancing**: Monitor per-expert load and adjust gating probabilities

### Communication Overlap and Scheduling
- **Compute-communication overlap**: Use CUDA streams and NCCL for asynchronous operations
- **Pipeline scheduling**: Each MoE layer as micro-stage with immediate token routing
- **Double buffering**: Buffer A for compute while Buffer B for communication

### Large EP Regime (EP ≥ 16)
- **Network optimization**: Topology-aware routing and token batching
- **Memory integration**: Optional tensor parallelism within expert if needed
- **Scalability**: Near-linear scaling with abundant GPU resources

## Experiments

### Setup
- **Model**: 4-layer MoE, 16 experts per layer (64 total), MLP experts
- **Precision**: FP16
- **Batch size**: 1024 tokens
- **Dimensions**: 16 attention heads × 512 dim/head = 8192 total, MLP hidden = 32768
- **Hardware**: H100 GPUs

### Baseline vs Proposed
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed Cross-Node | 64 | 1 expert per GPU | 450,000 | 2.2ms |

### Results
- **3.75× higher throughput** (450k vs 120k TPS)
- **3.8× lower latency** (2.2ms vs 8.3ms TPOT)
- **Near-linear scaling** with 64 GPUs
- **93.75% scaling efficiency** (3.75× improvement with 4× GPUs)

## Conclusion
Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU. By shifting the bottleneck from intra-GPU contention to communication (effectively mitigated through asynchronous routing and overlap), we achieve 3.75× higher throughput and 3.8× lower latency compared to traditional approaches. This provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.

## Key Technical Specifications
- **Model**: 4-layer MoE, 64 experts total
- **Expert memory**: ~1.07GB per expert (FP16)
- **Per-GPU memory**: 4.35GB (4 experts × 1.07GB + activations)
- **Compute**: 536.9 GFLOPs per expert forward pass
- **Communication**: 16MB token exchange per expert, 59% overlap achieved
- **Deployment**: 64 H100 GPUs, NCCL 2.18+, CUDA 12.0+