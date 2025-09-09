# Layer-wise Deployment for Large Neural Networks: A Concise Summary

### Abstract

In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## 1. Problem and Solution

**Challenge**: Large neural networks exceed on-chip memory capacity, forcing expensive off-chip DRAM accesses that create latency bottlenecks.

**Solution**: Layer-wise partitioning that splits *n* layers into *k* groups where each group fits in SRAM/L2 cache capacity *C*.

**Constraints**: 
- S(Pᵢ) = Σ size(lⱼ) ≤ C for all partitions
- Contiguous layer assignment preserving execution order
- Minimize *k* for balanced hardware utilization

## 2. Methodology

### Memory Footprint Estimation
Each layer size: size(lⱼ) = weight_size + activation_size + buffer_size
- **Weights**: parameters × datatype (FP16 = 2 bytes)
- **Activations**: output_feature_map × batch_size  
- **Buffers**: operator workspace from profiling

### Partitioning Algorithms

**Greedy Layer Aggregation**:
1. Initialize empty partition Pᵢ
2. Add layers sequentially while Σ size(lⱼ) ≤ C
3. When adding layer lⱼ exceeds C, finalize Pᵢ and start new partition
4. Continue until all layers assigned

**Dynamic Programming** (optional): Minimize maximum partition size for more balanced load.

### Deployment Process
1. **Pre-deployment**: Statically estimate layer sizes
2. **Partitioning**: Apply greedy algorithm to determine boundaries  
3. **Loading**: Load weights and pre-allocate memory within SRAM/L2 cache
4. **Execution**: Sequential layer execution on assigned card
5. **Communication**: Transfer intermediate outputs only between partitions on different cards

## 3. Experimental Results

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Model**: 16-layer dense network
- **Configuration**: FP16 precision, batch_size=1024
- **Architecture**: 16 heads, 512 head_dim, hidden_size=8192, MLP_hidden=32768

### Memory Requirements
- **Per layer**: 901 MB total
  - Weights: 805 MB
  - Activations: 16 MB  
  - Buffers: 80 MB
- **Cache capacity required**: ≥901 MB per partition

### Performance Comparison

| Method | GPUs | TPS (tokens/s) | TPOT (ms) | Improvement |
|--------|------|----------------|-----------|-------------|
| Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 | - |
| Proposed Layer-wise | 16 | 15,360 | 0.065 | +20% TPS, -17% TPOT |

### Key Benefits
- **20% throughput increase** from reduced off-chip memory access
- **17% latency reduction** due to cache locality
- **Scalable** to varying model sizes and hardware configurations
- **Minimal communication** overhead (only between layer boundaries)

## 4. Deployment Configuration Summary

**Baseline**: Hybrid tensor+pipeline parallelism (TP=8, PP=2) across 16 GPUs
**Proposed**: 16 layer-wise partitions, 1 layer per GPU, each fitting in 901 MB cache
**Hardware**: 16× NVIDIA H100 with ≥901 MB SRAM/L2 cache per device
**Precision**: FP16 with 1024 batch size for optimal memory utilization