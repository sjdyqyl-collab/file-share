# Phase One: Keypoints of the Paper

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as “large EP” in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Definitions
- **Large EP**: Expert Parallelism (EP) ≥ 16, prioritizing one expert per GPU to maximize concurrency.
- **MoE**: Mixture-of-Experts architecture where FFN layers are replaced by specialized experts, with a gating mechanism to select active experts per token.

## Core Method Components
1. **Expert Placement**: One expert per GPU, distributed across nodes using topology-aware routing to balance load.
2. **Routing/Load Balancing**: Token batching, asynchronous routing, and dynamic gating to avoid expert overloading.
3. **Communication Overlap**: Interleave token transfer (via NCCL/MPI) with GPU computation to minimize latency.

## Experimental Setup
- **Model**: 4-layer MoE, 16 experts/layer, FP16 precision.
- **Input**: 1024 sequences/batch, 10000 tokens/sequence, 8192 token dimension.
- **Baseline**: 16 GPUs, TP=8, PP=2, 4 experts/GPU.
- **Proposed**: 64 GPUs, 1 expert/GPU, asynchronous routing.

## Key Results
- **Throughput (TPS)**: Proposed (450k) vs. Baseline (120k) → 3.75× improvement.
- **Latency (TPOT)**: Proposed (2.2ms) vs. Baseline (8.3ms) → 3.8× reduction.

## Main Contributions
- Large EP definition and deployment strategy (one expert/GPU).
- Cross-node expert distribution to maximize compute concurrency.
- Asynchronous communication overlap to mitigate network bottlenecks.

## Conclusion
The method achieves near-linear scaling in large GPU clusters by shifting bottlenecks from intra-GPU contention to manageable communication, enabling high-throughput/low-latency MoE inference.