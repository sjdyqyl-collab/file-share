# Phase 1: Key Points Extraction

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Problem Statement
Traditional MoE parallelization assigns multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks and limiting expert parallelism as cluster sizes grow.

## Core Innovation
- **Single-Expert-Per-GPU Deployment**: Each GPU hosts at most one expert
- **Large Expert Parallelism (EP ≥ 16)**: Maximizes expert-level parallelism
- **Cross-Node Distribution**: Exploits distributed resources across nodes
- **Communication-Compute Overlap**: Mitigates network latency through asynchronous routing

## Technical Approach
1. **Expert Placement Strategy**: One expert per GPU, distributed across nodes
2. **Routing and Load Balancing**: Token batching and asynchronous routing
3. **Communication Overlap**: Interleaving computation and communication
4. **Scalability**: Optimized for EP ≥ 16 regime

## Experimental Results
- **Model**: 4-layer MoE, 16 experts per layer, MLP experts
- **Precision**: FP16
- **Batch**: 1024 sequences × 10000 tokens
- **Token Dimension**: 8192
- **Results**: 3.75× higher throughput, 3.8× lower latency vs baseline
- **Baseline**: TP=8, PP=2 with 16 GPUs (4 experts per GPU)
- **Proposed**: 64 GPUs (1 expert per GPU)