# Phase 1: Key Points of the Paper

## Core Problem
- Large neural network models face memory access bottlenecks due to limited on-chip memory (SRAM/L2 cache)
- Off-chip memory access introduces significant latency and bandwidth limitations
- Traditional parallelism methods don't explicitly consider on-chip memory constraints

## Key Innovation
- Novel layer-wise deployment strategy that partitions model layers across multiple accelerator cards
- Each partition must fit entirely within the SRAM or L2 cache of a single device
- Explicitly considers on-chip memory capacity constraints during partitioning

## Main Objectives
1. Minimize off-chip memory accesses
2. Maximize fast on-chip memory utilization
3. Preserve model execution order
4. Achieve scalable deployment across multiple devices

## Key Technical Components
- Memory footprint estimation formula: size(l_j) = weight_size(l_j) + activation_size(l_j) + buffer_size(l_j)
- Partitioning constraint: S(P_i) = sum_{l_j in P_i} size(l_j) ≤ C (cache capacity)
- Greedy layer aggregation algorithm for partitioning
- Optional dynamic programming approach for balanced partitions

## Performance Claims
- 20% increase in TPS (tokens per second) for dense model
- 17% reduction in TPOT (time per output token) for dense model
- Significant improvement over baseline TP=8, PP=2 configuration

## Hardware Context
- 16 NVIDIA H100 GPUs used in experiments
- FP16 precision with batch size 1024
- Dense 16-layer network as test case
- Cache capacity C is the limiting factor for partition sizes

## Critical Dimensions
- Model: 16 layers total
- Batch size: 1024
- Precision: FP16 (2 bytes per parameter)
- Hidden size: 8192 (16 heads × 512 dimensions per head)
- MLP hidden size: 32768
- Cache capacity C: must accommodate weights + activations + buffers for each partition