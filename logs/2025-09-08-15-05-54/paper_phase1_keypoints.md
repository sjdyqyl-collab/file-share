# Phase 1: Key Points of Large-Scale Cross-Node Expert Parallelism for MoE Models

## Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Problem
- Traditional MoE implementations colocate multiple experts on same GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- Trade-off becomes suboptimal as model and cluster sizes grow

## Core Innovation
- **Large Expert Parallelism (EP ≥ 16)**: Deploy at most one expert per GPU
- Shift bottleneck from intra-GPU contention to network communication
- Fully exploit distributed resources for maximum compute concurrency
- Leverage modern HPC networking (NVLink, InfiniBand, NVSwitch) to handle communication overhead

## Technical Components
1. **Expert Placement Strategy**: One expert per GPU, cross-node distribution
2. **Routing and Load Balancing**: Dynamic gating with balanced token distribution
3. **Communication Overlap**: Asynchronous token routing with compute-communication overlap
4. **Scalability**: Optimized for EP ≥ 16 with topology-aware placement

## Key Benefits
- Maximized expert parallelism with minimal contention
- Near-linear scaling in large EP regime
- 3.75× higher throughput and 3.8× lower latency vs baseline
- Compatible with tensor parallelism (TP) and data parallelism (DP) for large models

## Experimental Validation
- Model: 4-layer MoE, 16 experts per layer, MLP experts
- Precision: FP16
- Batch: 1024 sequences × 10000 tokens
- Token dimension: 8192
- MHA: 16 heads × 512 dim per head
- MLP hidden: 32768
- Results: 450,000 TPS vs 120,000 TPS baseline (3.75× improvement)