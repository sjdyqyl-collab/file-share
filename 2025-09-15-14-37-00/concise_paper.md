# Layer-wise Deployment for Large Neural Networks: A Concise Version

### Abstract
In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## 1. Introduction
Large deep learning models face deployment challenges due to memory hierarchy limitations. Traditional approaches often require off-chip memory access, introducing latency bottlenecks. We propose a layer-wise partitioning method that ensures each partition fits entirely within SRAM/L2 cache, minimizing memory access overhead while preserving model execution order.

## 2. Methodology

### 2.1 Problem Formulation
Given *n* layers $L = {l_1, l_2, ..., l_n}$, partition into $k$ groups $P = {P_1, P_2, ..., P_k}$ such that:
- $S(P_i) = \sum_{l_j \in P_i} \text{size}(l_j) \leq C$ (cache capacity)
- Layers assigned contiguously in original order
- Minimize $k$ for optimal hardware utilization

### 2.2 Memory Footprint Estimation
$$\text{size}(l_j) = \text{weight_size}(l_j) + \text{activation_size}(l_j) + \text{buffer_size}(l_j)$$

**Components:**
- **Weights**: Parameters × datatype (FP16=2 bytes)
- **Activations**: Output feature map × batch size
- **Buffers**: Operator workspace from profiling

### 2.3 Partitioning Algorithms

**Greedy Layer Aggregation:**
1. Initialize empty partition $P_i$
2. Add layers sequentially until $S(P_i) > C$
3. Finalize $P_i$, start new partition
4. Repeat until all layers assigned

**Dynamic Programming (Optional):**
- Minimize maximum partition size
- Achieve balanced load distribution

### 2.4 Deployment Strategy
- **Pre-deployment**: Calculate layer sizes, apply partitioning algorithm
- **Runtime**: Load weights + allocate memory in cache, execute sequentially
- **Communication**: Transfer outputs only between partitions on different cards

## 3. Experiments

### 3.1 Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Model**: 16-layer dense network
- **Configuration**: FP16, batch=1024, seq_len=10000, heads=16, head_dim=512, MLP_hidden=32768
- **Baseline**: TP=8, PP=2 (16 GPUs total)

### 3.2 Results
| Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|--------|------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| **Proposed Layer-wise** | 16 | **15,360** | **0.065** |

**Improvements:**
- **20% increase in TPS** (12,800 → 15,360)
- **17% reduction in TPOT** (0.078 → 0.065 ms)

### 3.3 Analysis
The proposed method achieves superior performance by:
- Fitting partitions entirely in SRAM/L2 cache
- Minimizing off-chip memory accesses
- Reducing communication overhead between partitions
- Maximizing computational efficiency per device

## 4. Conclusion
Our layer-wise deployment strategy demonstrates significant performance gains by explicitly considering on-chip memory constraints. The approach achieves 20% throughput improvement and 17% latency reduction over traditional tensor and pipeline parallelism, validating the effectiveness of cache-aware model partitioning for large-scale neural networks.

## 5. Technical Specifications Summary

### Model Parameters
- **Layers**: 16
- **Precision**: FP16 (2 bytes/param)
- **Batch size**: 1024
- **Sequence length**: 10000
- **Attention**: 16 heads × 512 dim = 8192 total
- **MLP**: 32768 hidden units

### Memory Calculation Example
For a dense layer with:
- Input: 8192, Output: 8192
- Weights: 8192 × 8192 × 2 bytes = 134.2 MB
- Activations: 8192 × 1024 × 2 bytes = 16.8 MB
- Buffers: ~1-5 MB (operator dependent)
- **Total per layer**: ~152 MB

### Cache Capacity Constraint
With 16 layers and 16 GPUs, optimal partition would be 1 layer per GPU (if 152 MB ≤ cache capacity), or adjust based on actual cache size using the partitioning algorithm.