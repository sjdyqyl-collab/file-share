# Phase 1: Key Points Extraction

## Problem Statement
- Large neural networks face deployment challenges due to limited on-chip memory (SRAM/L2 cache)
- External memory access introduces latency and bandwidth bottlenecks
- Need for deployment strategies that maximize fast on-chip memory utilization

## Core Contribution
- Novel layer-wise deployment strategy for large-scale neural networks
- Partitions n layers across multiple accelerator cards ensuring each partition fits entirely within SRAM/L2 cache
- Minimizes off-chip memory accesses and reduces latency

## Key Technical Details
- Memory footprint estimation includes: weights + activations + temporary buffers
- Partitioning algorithm: greedy layer aggregation or dynamic programming for balanced partitions
- Constraint: S(P_i) ≤ C where C is SRAM/L2 cache capacity
- Preserves execution order with contiguous layer assignment
- Minimizes inter-card communication by transferring only between partitions

## Performance Gains
- Dense 16-layer model: 20% increase in TPS (15,360 vs 12,800), 17% reduction in TPOT (0.065ms vs 0.078ms)
- Baseline comparison: TP=8, PP=2 on 16 NVIDIA H100 GPUs
- FP16 precision, batch size 1024

## Model Specifications
- Dense model: 16 layers, 16 heads, 512 head dimension, 32768 MLP hidden size