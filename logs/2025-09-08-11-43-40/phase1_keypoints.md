# Phase 1: Key Points Extraction

## Key Contributions
1. **Large-Scale Cross-Node Expert Parallelism**: Novel strategy for MoE models that deploys at most one expert per GPU
2. **Large EP Regime**: Defines "large EP" as EP ≥ 16, maximizing expert independence and computational parallelism
3. **Performance Gains**: Achieves 3.75× higher throughput and 3.8× lower latency compared to traditional approaches

## Core Problem Addressed
- Traditional MoE implementations colocate multiple experts on same GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism as model/cluster sizes grow
- Trade-off becomes increasingly suboptimal with modern HPC networking capabilities

## Key Innovation
- Shifts optimization focus from reducing communication to maximizing compute concurrency
- Leverages modern HPC networking (NVLink, InfiniBand, NVSwitch) to sustain high bandwidth/low latency
- Prioritizes distributing experts across nodes with one-expert-per-GPU principle

## Technical Highlights
- **Expert Placement**: Topology-aware distribution considering bandwidth, latency, GPU memory, routing patterns
- **Load Balancing**: Dynamic gating with token batching and asynchronous routing
- **Communication Overlap**: Interleaving expert computation with cross-node token transfers
- **Scalability**: Optimized for EP ≥ 16 with near-linear scaling

## Experimental Validation
- **Model**: 4-layer MoE, 16 experts per layer, MLP experts
- **Precision**: FP16
- **Scale**: 1024 sequences × 10000 tokens = 10.24M tokens per batch
- **Hardware**: H100 GPUs (16 vs 64 comparison)
- **Results**: 450,000 TPS vs 120,000 TPS baseline, 2.2ms vs 8.3ms TPOT

## Deployment Impact
- Baseline: 16 GPUs with 4 experts per GPU + TP=8, PP=2
- Proposed: 64 GPUs with 1 expert per GPU, full expert parallelism
- Demonstrates practical scalability blueprint for HPC environments