# Phase 1: Key Points Extraction

## Abstract
In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## Key Points Summary

### Core Problem
- Large neural network models (n layers) need efficient deployment across multiple processing units
- On-chip memory (SRAM/L2 cache) is limited but provides fast access
- Off-chip memory (DRAM) is abundant but slow, creating bottlenecks

### Proposed Solution
- **Layer-wise partitioning**: Split n layers into k disjoint groups
- **Cache constraint**: Each partition must fit entirely within SRAM/L2 cache capacity C
- **Contiguous allocation**: Preserve layer execution order
- **Memory optimization**: Minimize off-chip memory accesses

### Technical Approach
1. **Memory Footprint Estimation**: Calculate size(l_j) = weight_size + activation_size + buffer_size
2. **Partitioning Algorithm**: 
   - Greedy layer aggregation (simple, efficient)
   - Dynamic programming for balanced partitions (optional)
3. **Deployment Strategy**: Load each partition into SRAM/L2 cache of separate accelerator cards

### Performance Metrics
- **Tokens Per Second (TPS)**: Output tokens generated per second
- **Time Per Output Token (TPOT)**: Average time per token in milliseconds

### Experimental Results
- **Hardware**: 16 NVIDIA H100 GPUs
- **Model**: 16-layer dense network with FP16 precision
- **Batch size**: 1024
- **Improvement**: 20% increase in TPS (12,800 → 15,360), 17% reduction in TPOT (0.078ms → 0.065ms)
- **Baseline**: TP=8, PP=2 (tensor parallelism + pipeline parallelism)

### Key Advantages
- Reduced memory access latency
- Improved throughput via parallel execution
- Scalability across varying model sizes and hardware
- Maximizes on-chip memory utilization