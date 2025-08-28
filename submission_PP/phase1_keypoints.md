# Phase 1: Key Points Extraction

## Abstract
In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## Key Points

### Problem Statement
- Large neural network models face deployment challenges due to limited on-chip memory (SRAM/L2 cache)
- Off-chip memory access introduces latency and bandwidth bottlenecks
- Need for deployment strategies that leverage fast but limited on-chip memory

### Proposed Solution
- Layer-wise partitioning and distribution method for large models
- Split *n* layers across multiple accelerator cards
- Ensure each layer group fits entirely into SRAM/L2 cache
- Minimize memory access overhead and improve throughput

### Core Innovation
- Explicit consideration of SRAM/L2 cache size constraints during layer partitioning
- Systematic method to estimate memory footprint of each partition
- Dynamic allocation to hardware resources
- Preserves memory locality and efficiency

### Technical Approach
- Partition model layers into k disjoint groups
- Each group assigned to separate accelerator card
- Memory footprint constraint: S(P_i) ≤ C (cache capacity)
- Preserve execution order with contiguous layer assignment
- Minimize number of partitions k

### Memory Footprint Components
- Weights: parameter tensors
- Activations: intermediate outputs
- Temporary buffers: workspace memory for operators

### Performance Gains
- Dense model: 20% TPS increase, 17% TPOT reduction
- MoE model: 31% TPS increase, 23% TPOT reduction
- Reduced memory access latency through on-chip cache utilization
- Improved throughput via parallel execution
- Better scalability for varying model sizes and hardware configurations