# Phase 1: Key Points Extraction

## Abstract (Retained in full)
In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## Key Points

### Problem Statement
- Large neural networks face memory access bottlenecks due to limited on-chip SRAM/L2 cache
- Off-chip memory access introduces latency and bandwidth issues
- Need deployment strategies that maximize fast memory utilization

### Core Innovation
- **Layer-wise partitioning**: Split n layers into k groups P = {P₁, P₂, ..., Pₖ}
- **Cache constraint**: Each partition Pᵢ must fit within SRAM/L2 cache capacity C
- **Memory optimization**: Minimize off-chip memory accesses by keeping data in fast memory

### Memory Footprint Components
1. **Weights**: Parameter tensors (FP16 = 2 bytes)
2. **Activations**: Intermediate outputs (depends on batch size and dimensions)
3. **Temporary Buffers**: Workspace for operators

### Partitioning Algorithms
- **Greedy approach**: Sequential layer grouping until cache limit reached
- **Dynamic programming**: Balanced partitions to minimize k and maximize utilization

### Deployment Process
1. Partition layers respecting cache capacity C
2. Load each partition Pᵢ onto separate accelerator card
3. Execute sequentially within each card
4. Transfer data only between partitions on different cards

### Experimental Setup
- Hardware: 16 NVIDIA H100 GPUs
- Models: 16-layer dense network
- Precision: FP16
- Batch size: 1024
- Model dimensions: 16 heads, 512 head dimension, 32768 MLP hidden size
- Baseline: TP=8, PP=2 (tensor parallelism × pipeline parallelism)

### Performance Results
- Dense model: 20% TPS improvement (12,800 → 15,360 tokens/s)
- Dense model: 17% latency reduction (0.078 → 0.065 ms per token)
- Metric: Tokens Per Second (TPS) and Time Per Output Token (TPOT)

### Advantages
- Reduced memory access latency
- Improved throughput via parallel execution
- Scalable to varying model sizes and hardware configurations
- Explicit cache-aware optimization