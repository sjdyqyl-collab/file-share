### Abstract

In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

---

## Methodology

### Problem Formulation
Given a model with n layers L = {l₁, l₂, ..., lₙ}, partition into k disjoint groups P = {P₁, P₂, ..., Pₖ} where:
- Each Pᵢ assigned to separate accelerator card
- Memory footprint S(Pᵢ) ≤ C (cache capacity)
- Layers assigned contiguously in original order
- Minimize number of partitions k

**Constraint**: S(Pᵢ) = Σ_{lⱼ ∈ Pᵢ} size(lⱼ) ≤ C

### Memory Footprint Estimation
**size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)**
- weight_size: parameters × datatype (FP16 = 2 bytes)
- activation_size: output dimensions × batch size
- buffer_size: operator workspace requirements

### Partitioning Algorithms
1. **Greedy Layer Aggregation**: Sequential grouping until capacity reached
2. **Dynamic Programming**: Optimize for balanced partitions (optional)

### Deployment Strategy
- Load weights and pre-allocate memory within SRAM/L2 cache
- Execute layers sequentially on assigned card
- Minimize inter-card communication

## Experiments

### Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Model**: 16-layer dense network
- **Precision**: FP16, batch size 1024
- **Config**: 16 heads, 512 head dim, 32768 MLP hidden size
- **Baseline**: TP=8, PP=2 (tensor × pipeline parallelism)

### Results
| Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|--------|------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Proposed Layer-wise | 16 | 15,360 | 0.065 |

### Performance Gain
- **20% TPS increase** (12,800 → 15,360)
- **17% TPOT reduction** (0.078 → 0.065 ms)
- Improvement from reduced memory access latency via on-chip cache utilization

## Conclusion
Proposed layer-wise deployment significantly improves inference efficiency by ensuring each partition fits within on-chip cache, reducing off-chip memory accesses and achieving 20% throughput improvement over baseline approaches. Experimental results on a 16-layer dense model demonstrate substantial performance gains, with 20% improvement in throughput (TPS) and corresponding reduction in latency (TPOT).