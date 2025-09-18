# Paper Keypoints - Phase 1

## Abstract
In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## Key Problem
- Large neural network models exceed on-chip memory capacity (SRAM/L2 cache)
- Off-chip memory access causes latency and bandwidth bottlenecks
- Need deployment strategies that maximize fast memory utilization

## Core Contribution
- Layer-wise partitioning strategy for large models
- Ensures each partition fits entirely within SRAM/L2 cache of target device
- Minimizes off-chip memory accesses and improves throughput

## Key Technical Details
- Model with *n* layers partitioned into *k* disjoint groups
- Each group assigned to separate accelerator card
- Memory footprint constraint: S(P_i) ≤ C (cache capacity)
- Memory footprint includes: weights + activations + temporary buffers
- Uses greedy or dynamic programming partitioning algorithms

## Experimental Results
- Hardware: 16 NVIDIA H100 GPUs
- Model: 16-layer dense network, FP16, batch size 1024
- Baseline: TP=8, PP=2 configuration
- Results: 20% increase in TPS (12,800 → 15,360 tokens/s), 17% reduction in TPOT (0.078 → 0.065 ms)

## Key Dimensions
- Dense model: 16 layers
- Batch size: 1024
- Hidden size: 512 × 16 heads = 8192
- MLP hidden size: 32768
- Precision: FP16 (2 bytes per parameter)