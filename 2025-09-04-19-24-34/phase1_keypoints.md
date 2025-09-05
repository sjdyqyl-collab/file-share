# Phase 1: Key Points Extraction - Large-Scale Cross-Node Expert Parallelism for MoE Models

## Core Problem
Traditional MoE parallelization assigns multiple experts per GPU to reduce communication, creating computational bottlenecks and limiting expert-level parallelism as clusters scale.

## Key Innovation
Large-scale cross-node expert parallelism strategy that:
- Deploys at most **one expert per GPU**
- Achieves **Expert Parallelism (EP) ≥ 16** (defined as "large EP")
- Maximizes compute concurrency by shifting bottleneck from contention to communication
- Exploits modern HPC networking capabilities (NVLink, InfiniBand, NVSwitch)

## Technical Approach
1. **Expert Placement**: One-expert-per-GPU deployment, topology-aware distribution across nodes
2. **Routing**: Token batching, asynchronous routing, dynamic load balancing
3. **Communication**: Overlap computation with communication using CUDA streams/NCCL
4. **Scalability**: Optimized for EP ≥ 16 regime with network bandwidth as primary limiter

## Model Architecture
- 4-layer MoE with 16 experts per layer
- Each expert: MLP with hidden size 32768
- Precision: FP16
- Batch size: 1024 tokens
- MHA: 16 heads × 512 dimensions per head

## Performance Results
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert per GPU | 450,000 | 2.2ms |

**Improvements**: 3.75× higher throughput, 3.8× lower latency

## Deployment Requirements
- 64 H100 GPUs minimum for full deployment
- High-bandwidth interconnects (NVLink/InfiniBand)
- Asynchronous communication libraries (NCCL/MPI)
- Topology-aware placement algorithms

## Key Limitations
- Requires abundant GPU resources (64 GPUs for demonstrated configuration)
- Communication becomes primary bottleneck
- Currently inference-only (training extension is future work)
- Network topology significantly impacts performance