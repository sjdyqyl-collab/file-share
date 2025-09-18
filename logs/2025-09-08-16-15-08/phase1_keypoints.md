# Phase 1: Key Points Extraction

## Main Problem Addressed
Traditional MoE parallelization strategies assign multiple experts to the same GPU to reduce inter-node communication, which creates computational bottlenecks and limits expert-level parallelism as model and cluster sizes grow.

## Proposed Solution
A large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models that maximizes computational parallelism by deploying at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond.

## Key Innovations
1. **Single-Expert-Per-GPU Deployment**: Ensures each expert processes tokens without contention from other experts on the same device
2. **Cross-Node Distribution**: Topology-aware placement strategy considering node-to-node bandwidth, GPU memory capacity, and token routing patterns
3. **Asynchronous Token Routing**: Overlaps communication with computation using CUDA streams or NCCL/MPI
4. **Large EP Regime**: Optimized for EP ≥ 16, shifting bottleneck from communication to synchronization and load balancing

## Model Architecture
- 4-layer Mixture-of-Experts (MoE)
- 16 experts per layer
- Each expert is a MLP
- FP16 precision
- Token dimension: 8192
- Hidden size of MLP: 32768
- Batch size: 1024 sequences
- Sequence length: 10000 tokens per sequence

## Main Results
- **Baseline (TP=8, PP=2)**: 16 GPUs, 120,000 TPS, 8.3ms TPOT
- **Proposed Method**: 64 GPUs, 450,000 TPS, 2.2ms TPOT
- **Improvement**: ~3.75× higher throughput and ~3.8× lower latency

## Deployment Strategy
- Baseline: 16 H100 GPUs with 4 experts per GPU + TP shard
- Proposed: 64 H100 GPUs with 1 expert per GPU, fully utilizing all GPUs for expert-level parallelism