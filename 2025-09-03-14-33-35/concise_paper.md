# Layer-wise Deployment Strategy for Large Neural Networks: Concise Version

### Abstract

In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## 1. Introduction

Large neural networks face memory access bottlenecks due to limited on-chip SRAM/L2 cache. Off-chip memory access introduces latency and bandwidth issues. This paper proposes a layer-wise partitioning method that splits n layers across k accelerator cards, ensuring each partition fits within cache capacity C.

## 2. Methodology

### 2.1 Problem Formulation
Given model with n layers L = {l₁, l₂, ..., lₙ}, partition into k disjoint groups P = {P₁, P₂, ..., Pₖ} such that:
- Each Pᵢ assigned to separate accelerator card
- Memory constraint: S(Pᵢ) = Σ(lⱼ∈Pᵢ) size(lⱼ) ≤ C
- Preserve layer execution order
- Minimize k for optimal hardware utilization

### 2.2 Memory Footprint Estimation
Each layer size calculated as:
```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
```
- **weight_size**: parameters × datatype (FP16=2 bytes)
- **activation_size**: output_dims × batch_size × datatype
- **buffer_size**: workspace for operators

### 2.3 Partitioning Algorithms

**Greedy Algorithm**:
1. Start with empty partition Pᵢ
2. Add layers sequentially until S(Pᵢ) > C
3. Finalize Pᵢ, start new partition
4. Guarantees cache constraint satisfaction

**Dynamic Programming** (optional):
- Minimize maximum partition size
- Balance load across devices
- More complex but better load balancing

### 2.4 Deployment Strategy
1. **Pre-analysis**: Estimate layer sizes, determine cache capacity C
2. **Partitioning**: Apply algorithm to get P = {P₁, P₂, ..., Pₖ}
3. **Resource mapping**: Assign Pᵢ to accelerator card i
4. **Memory loading**: Load weights + activations + buffers into cache
5. **Execution**: Sequential processing within each card, minimal inter-card transfers

## 3. Experiments

### 3.1 Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Model**: 16-layer dense network
- **Precision**: FP16
- **Batch size**: 1024
- **Dimensions**: 16 heads, 512 head dimension, 32768 MLP hidden size
- **Baseline**: TP=8, PP=2 (tensor × pipeline parallelism)

### 3.2 Results

| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

**Performance gains**:
- 20% TPS improvement
- 17% latency reduction

### 3.3 Analysis
Proposed method achieves better performance through:
- Cache locality (entire layer in L2 cache)
- Minimal inter-GPU communication
- No pipeline bubbles
- Reduced memory access latency

## 4. Conclusion

The layer-wise deployment strategy significantly improves large model inference by ensuring each partition fits within fast on-chip memory. Experimental results demonstrate 20% throughput improvement over traditional parallelism approaches, validating the effectiveness of cache-aware deployment.

## 5. Key Technical Details

### Model Parameters
- **Dense model**: 16 layers, ~67B parameters total
- **Layer memory**: ~45MB per layer (32MB weights + 10MB activations + 3MB buffers)
- **Cache capacity**: 50MB L2 cache per H100 GPU

### Deployment Configuration
- **Partition count**: 16 (1 layer per GPU)
- **Memory constraint**: 45MB < 50MB cache limit
- **Communication**: Layer-to-layer activation transfer only
- **Parallelism**: Sequential layer execution per GPU