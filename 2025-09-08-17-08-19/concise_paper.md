# Layer-wise Cache-Optimized Deployment for Large Neural Networks

### Abstract
In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## 1. Introduction
Large neural network models exceed on-chip memory capacity, causing off-chip memory access bottlenecks. We propose a layer-wise partitioning strategy that ensures each partition fits entirely within SRAM/L2 cache, minimizing memory access overhead and improving throughput.

## 2. Methodology

### 2.1 Problem Formulation
Given *n* layers L = {l₁, l₂, ..., lₙ}, partition into *k* disjoint groups P = {P₁, P₂, ..., Pₖ} such that:
- S(Pᵢ) ≤ C (cache capacity)
- Contiguous layer assignment
- Minimize *k* for hardware utilization

Where S(Pᵢ) = Σ weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)

### 2.2 Partitioning Algorithms
**Greedy Layer Aggregation:**
1. Initialize empty partition Pᵢ
2. Iteratively add layers until S(Pᵢ) > C
3. Finalize partition, start new one
4. Repeat until all layers assigned

**Dynamic Programming (optional):** Optimizes for balanced partitions

### 2.3 Deployment Strategy
- Load partition weights into cache
- Pre-allocate activations/buffers in cache
- Execute sequentially on assigned card
- Transfer outputs only between partitions

## 3. Experiments

### 3.1 Setup
- **Hardware:** 16× NVIDIA H100 GPUs
- **Model:** 16-layer dense network
- **Precision:** FP16
- **Batch Size:** 1024
- **Dimensions:** 16 heads × 512 = 8192 hidden, 32768 MLP hidden

### 3.2 Baseline Configuration
- Tensor Parallelism: TP=8
- Pipeline Parallelism: PP=2
- Total GPUs: 16 (8×2)

### 3.3 Results
| Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|--------|------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Proposed Layer-wise | 16 | 15,360 | 0.065 |

**Performance Gain:** 20% increase in TPS, 17% reduction in TPOT

## 4. Conclusion
Our cache-optimized layer-wise deployment achieves significant performance improvements by explicitly considering on-chip memory constraints, demonstrating 20% throughput increase over traditional tensor/pipeline parallelism approaches.

## Key Dimensions for Deployment
- Model: 16 layers
- Hidden size: 8192 (16×512)
- MLP hidden: 32768
- Batch size: 1024
- Precision: FP16 (2 bytes/parameter)
- Cache capacity: ~40MB per device (estimated)