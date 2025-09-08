# Phase 1: Key Points Extraction

## Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Problem
Traditional MoE parallelization assigns multiple experts per GPU to reduce communication, creating computational bottlenecks and limiting expert-level parallelism as model/cluster sizes grow.

## Core Solution
Large-scale cross-node expert parallelism with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond.

## Key Components
1. **Expert Placement Strategy**: One expert per GPU, distributed across nodes
2. **Routing and Load Balancing**: Dynamic gating with balanced input distribution
3. **Communication Overlap**: Asynchronous token routing with computation overlap

## Key Results
- 3.75× higher throughput (450,000 vs 120,000 TPS)
- 3.8× lower latency (2.2 vs 8.3 ms TPOT)
- 64 GPUs vs 16 GPUs baseline
- Large EP regime: EP ≥ 16

## Key Model Specs
- 4-layer MoE
- 16 experts per layer (64 total)
- FP16 precision
- 1024 sequences per batch
- 10,000 tokens per sequence
- 8192 token dimension
- 16 MHA heads, 512 head dimension
- 32,768 MLP hidden size

## Key Deployment Config
**Baseline**: TP=8, PP=2, 16 GPUs, 4 experts per GPU
**Proposed**: 64 GPUs, 1 expert per GPU, cross-node distribution