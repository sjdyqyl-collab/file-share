# Phase 1: Keypoints Extraction

## Problem Statement
The paper addresses the challenge of efficiently deploying large-scale neural network models on hardware with limited on-chip memory (SRAM/L2 cache). Large models require external memory access, creating latency and bandwidth bottlenecks.

## Proposed Solution
A novel layer-wise deployment strategy that partitions model layers across multiple accelerator cards while ensuring each partition fits entirely within the SRAM or L2 cache of a single device.

## Key Technical Components
1. **Memory Footprint Estimation**: Calculates layer memory usage including weights, activations, and temporary buffers
2. **Partitioning Algorithm**: Greedy or dynamic programming approach to divide n layers into k disjoint groups
3. **Deployment Strategy**: Maps each partition to separate accelerator cards with pre-allocated memory
4. **Constraint Satisfaction**: Ensures each partition size S(Pi) ≤ cache capacity C

## Experimental Setup
- Hardware: 16 NVIDIA H100 GPUs
- Models: 16-layer dense network
- Precision: FP16
- Batch size: 1024
- Baseline: Tensor parallelism (TP=8) + Pipeline parallelism (PP=2)

## Key Results
- Dense model: 20% increase in TPS (15,360 vs 12,800), 17% reduction in TPOT (0.065ms vs 0.078ms)
- Performance improvement attributed to reduced off-chip memory access

## Advantages
- Reduced memory access latency
- Improved throughput
- Scalability across different model sizes and hardware configurations

## Future Work
- Extension to training workloads
- Adaptive partitioning for varying batch sizes
- Application to larger, more complex models