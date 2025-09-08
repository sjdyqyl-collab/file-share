# Phase 1: Key Points Extraction

## Core Problem
Traditional MoE implementations colocate multiple experts on GPUs, creating computational bottlenecks and limiting expert-level parallelism as models and clusters grow.

## Proposed Solution
Large-scale cross-node expert parallelism strategy that:
- Deploys at most one expert per GPU
- Achieves Expert Parallelism (EP) ≥ 16 ("large EP")
- Maximizes computational parallelism by exploiting distributed resources
- Reduces expert-level contention and improves throughput

## Key Innovations
1. **Single-Expert-Per-GPU Deployment**: Ensures minimal contention and high compute efficiency
2. **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory
3. **Asynchronous Token Routing**: Overlaps computation with communication
4. **Dynamic Load Balancing**: Prevents network bottlenecks and expert overloading

## Performance Gains
- **3.75× higher throughput** (450,000 vs 120,000 TPS)
- **3.8× lower latency** (2.2ms vs 8.3ms TPOT)
- Near-linear scaling in large EP regime (EP ≥ 16)

## Technical Specifications
- Model: 4-layer MoE, 16 experts per layer
- Precision: FP16
- Batch: 1024 sequences × 10000 tokens
- Token dimension: 8192
- MHA: 16 heads × 512 dimensions
- MLP hidden size: 32768
- Deployment: 64 H100 GPUs (1 expert per GPU)

## Scalability Features
- Compatible with tensor parallelism (TP) and data parallelism (DP)
- Handles models exceeding single-GPU memory
- Optimized for HPC environments with high-bandwidth interconnects
- Supports future extensions to training scenarios and dynamic routing