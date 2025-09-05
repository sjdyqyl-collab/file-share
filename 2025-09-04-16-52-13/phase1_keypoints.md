# Phase 1: Key Points Extraction

## Core Problem Addressed
- Traditional MoE implementations colocate multiple experts on the same GPU, creating computational bottlenecks and limiting expert-level parallelism
- Scaling MoE models across large GPU clusters introduces challenges in expert placement and parallelization

## Key Innovation
- **Large-scale cross-node expert parallelism strategy** that deploys at most one expert per GPU
- **Large EP regime**: Expert Parallelism (EP) ≥ 16, where experts are distributed across as many devices as possible
- Shifts optimization focus from reducing communication to maximizing compute concurrency

## Main Contributions
1. **Single-expert-per-GPU deployment**: Eliminates intra-GPU contention between experts
2. **Cross-node expert distribution**: Topology-aware placement considering bandwidth, latency, and memory
3. **Asynchronous token routing**: Overlaps communication with computation
4. **Load balancing**: Dynamic gating adjustments to prevent expert overloading
5. **Scalability**: Optimized for high-performance computing environments with advanced networking

## Key Technical Components
- Expert placement strategy with topology awareness
- Token batching and asynchronous routing
- Pipeline scheduling for multi-layer MoE networks
- Integration with tensor parallelism and data parallelism when needed

## Performance Gains
- **3.75× higher throughput** (450,000 vs 120,000 TPS)
- **3.8× lower latency** (2.2ms vs 8.3ms TPOT)
- Near-linear scaling with 64 GPUs (EP=64)

## Critical Dimensions and Parameters
- Model: 4-layer MoE, 16 experts per layer
- Expert specification: MLP with hidden size 32768
- Precision: FP16
- Batch size: 1024 tokens per forward pass
- MHA: 16 heads × 512 dimensions per head
- Baseline: TP=8, PP=2 with 16 GPUs
- Proposed: EP=64 with 64 GPUs (1 expert per GPU)