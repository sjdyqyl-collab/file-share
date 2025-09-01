# Phase 1: Key Points Extraction

## Key Points from the Paper

### 1. Core Problem
- **Challenge**: Large neural network models exceed on-chip memory (SRAM/L2 cache) capacity, causing expensive off-chip memory accesses
- **Impact**: Memory access latency and bandwidth bottlenecks degrade system performance

### 2. Proposed Solution
- **Method**: Layer-wise partitioning and distribution strategy
- **Key Insight**: Ensure each partition fits entirely within SRAM/L2 cache of a single device
- **Objective**: Minimize off-chip data movement and maximize computational efficiency

### 3. Technical Approach
- **Partitioning**: Split n layers into k disjoint groups across multiple accelerator cards
- **Constraint**: Memory footprint of each group must not exceed cache capacity C
- **Preservation**: Maintain original layer execution order (contiguous assignment)

### 4. Memory Footprint Components
- **Weights**: Parameter tensors (FP16 = 2 bytes)
- **Activations**: Intermediate outputs (depends on batch size and feature dimensions)
- **Temporary Buffers**: Workspace for operators

### 5. Partitioning Algorithms
- **Greedy Layer Aggregation**: Simple sequential addition until capacity reached
- **Dynamic Programming**: Optional balanced partitioning to minimize k and balance load

### 6. Deployment Strategy
- **Per Card**: Load weights, pre-allocate activations/buffers within cache
- **Execution**: Sequential layer processing on assigned card
- **Communication**: Only transfer between partitions on different cards

### 7. Performance Results
- **Dense Model (16-layer)**: 20% TPS increase, 17% TPOT reduction vs baseline
- **MoE Model (16-layer, 8 experts)**: 31% TPS increase, 23% TPOT reduction vs baseline
- **Baseline**: TP=8, PP=2 on 16 NVIDIA H100 GPUs

### 8. Key Dimensions
- **Models**: 16 layers (dense and MoE)
- **MoE**: 8 experts per layer
- **Precision**: FP16
- **Batch Size**: 1024
- **Heads**: 16
- **Head Dimension**: 512
- **MLP Hidden Size**: 32768

### 9. Advantages
- **Reduced Memory Access Latency**: Minimal off-chip DRAM access
- **Improved Throughput**: Faster memory access + parallel execution
- **Scalability**: Adaptable to varying model sizes and hardware