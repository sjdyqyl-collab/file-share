# Layer-wise Deployment Strategy for Large Neural Networks - Key Points

## Abstract
In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## Key Points

### Problem Statement
- Large neural networks face memory access bottlenecks due to limited on-chip memory (SRAM/L2 cache)
- Current deployment strategies (tensor parallelism, pipeline parallelism) don't optimize for on-chip memory constraints
- Off-chip memory access introduces significant latency and bandwidth limitations

### Core Solution
- **Layer-wise partitioning**: Split model's *n* layers into *k* disjoint groups
- **Cache constraint**: Each partition must fit entirely within SRAM/L2 cache capacity *C*
- **Contiguous assignment**: Layers assigned in original order to preserve execution flow
- **Memory optimization**: Minimize off-chip access by maximizing on-chip memory utilization

### Technical Approach
1. **Memory footprint estimation** for each layer:
   - Weights: parameter tensors × datatype size
   - Activations: output feature maps × batch size
   - Temporary buffers: operator workspace requirements

2. **Partitioning algorithms**:
   - Greedy layer aggregation: sequentially add layers until cache limit reached
   - Dynamic programming: optimize for balanced partitions (optional)

3. **Deployment strategy**:
   - Each partition loaded entirely into on-chip memory
   - Sequential execution within each card
   - Minimal inter-card communication only between partitions

### Performance Results
- **Dense model (16-layer)**: 20% TPS improvement, 17% TPOT reduction
- **MoE model (16-layer, 8 experts)**: 31% TPS improvement, 23% TPOT reduction
- Hardware: 16 NVIDIA H100 GPUs, FP16 precision, batch size 1024
- Baseline comparison: TP=8, PP=2 configuration

### Key Advantages
- Reduced memory access latency through on-chip memory utilization
- Improved throughput via parallel execution across multiple cards
- Scalable to varying model sizes and hardware configurations
- Handles edge cases through compression techniques and batch size tuning