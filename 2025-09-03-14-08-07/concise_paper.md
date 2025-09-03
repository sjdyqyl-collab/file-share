# Layer-wise Deployment Strategy for Large Neural Networks: A Concise Version

### Abstract

In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## 1. Introduction and Background

Large neural networks face deployment challenges due to limited on-chip memory (SRAM/L2 cache), where external memory access introduces latency and bandwidth bottlenecks. Modern accelerators offer high computational throughput but limited on-chip memory capacity compared to external DRAM. Accessing on-chip SRAM/cache is significantly faster and more energy-efficient than off-chip memory.

We introduce a layer-wise partitioning and distribution method where *n* layers are split and mapped onto multiple accelerator cards, ensuring each layer group assigned to a card fits entirely into its SRAM or L2 cache. This minimizes memory access overhead and improves throughput during inference or training.

## 2. Methodology

### 2.1 Problem Formulation
Given model with *n* layers L = {l₁, l₂, ..., lₙ}, partition into *k* disjoint groups P = {P₁, P₂, ..., Pₖ} such that:
- Memory footprint S(Pᵢ) ≤ C (SRAM/L2 cache capacity)
- Full execution order preserved (contiguous layer assignment)
- Number of partitions *k* minimized or balanced

Formula: S(Pᵢ) = Σ_{lⱼ∈Pᵢ} size(lⱼ) ≤ C

### 2.2 Memory Footprint Estimation
Each layer size includes:
- **Weights**: Parameter tensors (datatype size × parameters)
- **Activations**: Intermediate outputs (output feature map dimensions × batch size)
- **Temporary Buffers**: Workspace memory for operators

Calculation: size(lⱼ) = weight_size(lⱼ) + activation_size(lⱼ) + buffer_size(lⱼ)

### 2.3 Partitioning Algorithms

#### Greedy Layer Aggregation
1. Start from layer l₁
2. Initialize empty partition Pᵢ
3. Iteratively add layers lⱼ to Pᵢ, accumulating S(Pᵢ)
4. If adding lⱼ causes S(Pᵢ) > C, finalize Pᵢ with layers {l_start, ..., lⱼ₋₁}
5. Start new partition Pᵢ₊₁ from layer lⱼ
6. Repeat until all layers assigned

#### Dynamic Programming (Optional)
Optimizes partition boundaries to minimize maximum partition size while respecting cache capacity.

### 2.4 Deployment Strategy
1. Assign each group Pᵢ to separate accelerator card
2. Load all weights and pre-allocate activation/buffer memory within SRAM/L2 cache
3. Execute layers sequentially on assigned card
4. Transfer intermediate outputs only between partitions on different cards

### 2.5 Edge Cases
- Single layer exceeding capacity C: use intra-layer partitioning or model compression
- Batch size tuning to reduce activation memory
- Variable layer sizes: adjust partitioning heuristics to avoid under-utilization

## 3. Experiments

### 3.1 Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Precision**: FP16
- **Batch size**: 1024
- **Model**: 16-layer fully connected dense network
- **Configuration**: 16 heads, 512 dimension per head, 32768 MLP hidden size

### 3.2 Baseline Comparison
- **Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- **Total GPUs**: 16 (8×2=16)

### 3.3 Results

| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

### 3.4 Analysis
- **20% increase in TPS** (15,360 vs 12,800)
- **17% reduction in TPOT** (0.065ms vs 0.078ms)
- Improvement attributed to efficient on-chip memory utilization
- Baseline doesn't explicitly consider on-chip memory constraints, leading to more off-chip memory accesses

## 4. Conclusion

Our layer-wise deployment strategy partitions model layers across multiple accelerator cards while ensuring each partition fits within SRAM/L2 cache constraints. This significantly reduces off-chip memory accesses and improves inference efficiency, achieving up to 20% improvement in throughput over baseline tensor and pipeline parallelism setups.