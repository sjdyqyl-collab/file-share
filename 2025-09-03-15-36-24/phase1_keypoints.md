# Phase 1: Key Points Extraction

## Core Problem
Traditional MoE implementations colocate multiple experts on the same GPU to reduce communication, creating computational bottlenecks and limiting expert-level parallelism.

## Proposed Solution
Large-scale cross-node expert parallelism strategy that:
- Deploys at most one expert per GPU
- Uses Expert Parallelism (EP) ≥ 16 (defined as "large EP")
- Maximizes computational parallelism by fully exploiting distributed resources

## Key Innovations

### 1. Expert Placement Strategy
- Single-expert-per-GPU deployment ensures minimal contention
- Cross-node distribution with topology-aware placement considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns

### 2. Routing and Load Balancing
- Token batching: Group tokens by destination expert
- Asynchronous routing: Send token batches asynchronously to overlap with computation
- Dynamic load balancing: Monitor per-expert load and adjust gating probabilities

### 3. Communication Overlap and Scheduling
- Overlap compute and communication using CUDA streams/NCCL/MPI
- Pipeline scheduling for multi-layer MoE networks
- Immediate routing between layers without waiting for full batch completion

## Model Configuration
- 4-layer MoE with 16 experts per layer
- Each expert is an MLP
- FP16 precision
- Batch size: 1024 tokens per forward pass
- MHA: 16 heads × 512 dimensions each
- MLP hidden size: 32768

## Performance Results
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert per GPU | 450,000 | 2.2ms |

## Scalability Features
- Optimized for EP ≥ 16 regime
- Network bandwidth as primary limiting factor
- Integration with tensor model parallelism (TP) for memory constraints
- Data parallelism (DP) for synchronized weight updates