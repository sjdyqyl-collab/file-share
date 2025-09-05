# Phase 1: Keypoints Extraction

## Problem Statement
Traditional MoE parallelization strategies colocate multiple experts on the same GPU to reduce inter-node communication, but this creates computational bottlenecks and limits expert-level parallelism as model and cluster sizes grow.

## Proposed Solution
A large-scale cross-node expert parallelism strategy that:
- Deploys at most one expert per GPU
- Distributes experts across nodes to fully exploit distributed resources
- Achieves Expert Parallelism (EP) ≥ 16 (defined as "large EP")

## Key Innovations
1. **Single-Expert-Per-GPU Deployment**: Each GPU hosts at most one expert, eliminating intra-GPU contention
2. **Cross-Node Distribution**: Topology-aware expert placement considering bandwidth, latency, and memory capacity
3. **Asynchronous Token Routing**: Overlapping communication with computation using token batching and asynchronous routing
4. **Load Balancing**: Dynamic gating probability adjustment to prevent expert overload
5. **Pipeline Scheduling**: Fine-grained pipeline for multi-layer MoE networks

## Technical Details
- Model: 4-layer MoE with 16 experts per layer
- Precision: FP16
- Batch size: 1024 sequences
- Sequence length: 10000 tokens
- MHA: 16 heads × 512 dimensions each
- MLP hidden size: 32768

## Main Results
- **3.75× higher throughput** (450,000 vs 120,000 tokens/second)
- **3.8× lower latency** (2.2 vs 8.3 ms per token)
- Uses 64 H100 GPUs vs 16 in baseline
- Near-linear scaling in large EP regime (EP ≥ 16)

## Deployment Comparison
| Method | GPUs | Per-GPU Deployment | TPS | TPOT |
|--------|------|-------------------|-----|------|
| Baseline | 16 | 4 experts + TP shard | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert per GPU | 450,000 | 2.2ms |