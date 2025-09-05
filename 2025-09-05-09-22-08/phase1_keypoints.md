# Phase 1: Key Points Extraction

## Core Problem
Traditional MoE implementations colocate multiple experts per GPU to reduce communication overhead, but this creates computational bottlenecks and limits expert-level parallelism as model and cluster sizes grow.

## Proposed Solution
Large-scale cross-node expert parallelism strategy that:
- Deploys at most one expert per GPU
- Distributes experts across nodes to maximize computational parallelism
- Targets Expert Parallelism (EP) ≥ 16 (defined as "large EP")

## Key Innovations
1. **Single-Expert-Per-GPU Deployment**: Each GPU hosts exactly one expert, eliminating intra-GPU contention
2. **Cross-Node Distribution**: Topology-aware placement considering node-to-node bandwidth, latency, GPU memory, and routing patterns
3. **Asynchronous Token Routing**: Overlapping computation and communication through token batching and asynchronous transfers
4. **Load Balancing**: Dynamic gating probability adjustment to prevent expert overloading

## Technical Details
- **Model Architecture**: 4-layer MoE, 16 experts per layer, each expert is an MLP
- **Precision**: FP16
- **Batch Size**: 1024 tokens per forward pass
- **MHA Configuration**: 16 heads, 512 dimensions per head
- **MLP Hidden Size**: 32768

## Performance Results
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert per GPU | 450,000 | 2.2ms |

## Improvements
- **3.75× higher throughput** (450k vs 120k TPS)
- **3.8× lower latency** (2.2ms vs 8.3ms TPOT)
- Near-linear scaling in large EP regime (EP ≥ 16)

## Scalability Considerations
- Network bandwidth becomes primary limiting factor in large EP regime
- Compatible with tensor model parallelism (TP) and data parallelism (DP)
- Suitable for HPC environments with high-bandwidth interconnects (NVLink, InfiniBand, NVSwitch)