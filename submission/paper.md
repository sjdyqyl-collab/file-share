# Layer-wise Deployment Strategy for Large Neural Networks

## Abstract
We propose a layer-wise deployment strategy that partitions neural network layers across multiple accelerators such that each partition fits entirely within SRAM/L2 cache, minimizing off-chip memory access. Our method achieves 20-31% throughput improvements over traditional tensor/pipeline parallelism.

## 1. Problem Statement
Large neural networks face memory access bottlenecks due to limited on-chip SRAM/L2 cache capacity. Traditional tensor and pipeline parallelism don't explicitly optimize for on-chip memory constraints, leading to frequent off-chip memory accesses that introduce significant latency.

## 2. Proposed Solution
**Layer-wise partitioning** that splits n layers into k groups P = {P₁, P₂, ..., Pₖ} where:
- Memory constraint: S(Pᵢ) = Σ size(lⱼ) ≤ C (cache capacity)
- Execution order: Layers assigned contiguously in original order
- Optimization: Minimize k while maximizing hardware utilization

## 3. Methodology

### 3.1 Memory Footprint Estimation
**Layer memory calculation:**
```
size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)
```

**Components:**
- **Weights**: Parameter tensors × datatype size (FP16 = 2 bytes)
- **Activations**: Output feature maps × batch size
- **Temporary Buffers**: Operator workspace from profiling

### 3.2 Partitioning Algorithms

**Greedy Layer Aggregation:**
1. Initialize empty partition Pᵢ
2. Iteratively add layers lⱼ to Pᵢ, accumulating S(Pᵢ)
3. If S(Pᵢ) > C, finalize Pᵢ with layers {lₛₜₐᵣₜ, ..., lⱼ₋₁}
4. Start new partition Pᵢ₊₁ from layer lⱼ
5. Repeat until all layers assigned

**Dynamic Programming (Optional):**
- Minimize maximum partition size while respecting cache constraints
- Balance load across partitions

### 3.3 Deployment Strategy
1. **Load phase**: Load weights and pre-allocate memory in SRAM/L2 cache
2. **Execution**: Sequential layer execution on assigned card
3. **Communication**: Transfer outputs only between partitions on different cards

## 4. Experiments

### 4.1 Experimental Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Models**: 
  - 16-layer dense network
  - 16-layer MoE (8 experts/layer)
- **Configuration**: FP16 precision, batch size 1024
- **Baseline**: Tensor parallelism=8, pipeline parallelism=2 (TP=8, PP=2)
- **Metrics**: Tokens Per Second (TPS), Time Per Output Token (TPOT)

### 4.2 Results

| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | **Proposed Layer-wise** | 16 | **15,360** | **0.065** |
| MoE (16-layer, 8 experts) | Baseline (TP=8, PP=2) | 16 | 10,200 | 0.098 |
| MoE (16-layer, 8 experts) | **Proposed Layer-wise** | 16 | **13,400** | **0.075** |

### 4.3 Analysis
- **Dense model**: 20% TPS improvement (12,800→15,360), 17% TPOT reduction
- **MoE model**: 31% TPS improvement (10,200→13,400), 23% TPOT reduction
- **Key insight**: Explicit cache-aware partitioning reduces off-chip memory accesses more effectively than traditional parallelism

## 5. Conclusion
Our layer-wise deployment strategy achieves significant performance gains (20-31% throughput improvement) by ensuring each model partition fits within on-chip cache, minimizing expensive off-chip memory accesses. This approach is particularly effective for complex models like MoE with irregular computation patterns.