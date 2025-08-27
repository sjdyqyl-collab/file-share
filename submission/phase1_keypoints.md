# Phase 1: Key Points Extraction

## Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Problem
Traditional MoE parallelization assigns multiple experts to the same GPU, creating computational bottlenecks and limiting expert-level parallelism as clusters grow.

## Core Innovation
Deploy at most one expert per GPU across nodes to maximize expert-level parallelism (EP ≥ 16), shifting bottleneck from intra-GPU contention to network communication.

## Technical Approach
1. **Expert Placement**: One expert per GPU, distributed across nodes with topology-aware placement
2. **Routing**: Dynamic token routing with asynchronous batching and load balancing
3. **Communication**: Overlapping computation and communication using CUDA streams/NCCL

## Key Benefits
- Maximized expert parallelism with minimal contention
- Balanced load across nodes
- Scalable communication overlap for EP ≥ 16
- Compatible with tensor and data parallelism

## Experimental Results
- Model: 4-layer MoE, 16 experts/layer, MLP experts
- Precision: FP16, Batch: 1024 tokens
- Baseline (TP=8, PP=2): 16 GPUs, 4 experts/GPU → 120k TPS, 8.3ms TPOT
- Proposed: 64 GPUs, 1 expert/GPU → 450k TPS, 2.2ms TPOT
- **3.75× higher throughput, 3.8× lower latency**