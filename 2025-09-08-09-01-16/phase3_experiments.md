# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Model Type**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16
- **Total experts**: 64
- **Expert Architecture**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16
- **Batch size**: 1024 sequences
- **Sequence length**: 10,000 tokens per sequence
- **Token dimension**: 8192
- **Multi-Head Attention**: 16 heads, 512 dimension per head
- **MLP hidden size**: 32,768

### 1.2 Hardware Environment
- **GPU Type**: H100 GPUs
- **Setting**: Inference-only

### 1.3 Metrics
- **TPS (Tokens per Second)**: Throughput measurement
- **TPOT (Time per Output Token)**: Latency per token measurement

## 2. Parallel Deployment Details

### 2.1 Baseline Deployment (TP=8, PP=2)
- **GPUs Used**: 16 H100
- **Parallel Configuration**:
  - Tensor Parallelism (TP): 8
  - Pipeline Parallelism (PP): 2
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (2 stages total)
  - Experts colocated: 4 experts per GPU
- **Processing**: Tokens flow sequentially through pipeline stages with shared compute resources

### 2.2 Proposed Cross-Node Expert Parallelism
- **GPUs Used**: 64 H100
- **Per-GPU Allocation**:
  - Each GPU hosts exactly one expert
  - Tensor parallelism: Applied only if single expert's FFN cannot fit (optional TP=2)
  - Pipeline parallelism: Each MoE layer as micro-stage with overlapped communication
- **Routing**:
  - Dynamic routing to GPU holding corresponding expert
  - Asynchronous token batch sending with minimal idle time
- **Expert Distribution**: All 64 experts per layer compute in parallel

## 3. Results Comparison

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### 3.1 Performance Improvements
- **Throughput**: 3.75× higher (450,000 vs 120,000 TPS)
- **Latency**: 3.8× lower (2.2 vs 8.3 ms TPOT)
- **Resource Scaling**: 4× more GPUs (64 vs 16) with near-linear scaling

### 3.2 Key Advantages
- **No intra-GPU contention** in proposed method
- **Full GPU compute utilization** with dedicated experts
- **Near-linear scaling** in large EP regime (EP ≥ 16)
- **Asynchronous routing** minimizes waiting across nodes

## 4. Discussion Points
- **Expert Isolation**: One expert per GPU eliminates resource sharing bottlenecks
- **Communication Overlap**: Asynchronous token routing ensures minimal idle time
- **Scalability**: With 64 GPUs, system scales near-linearly in large EP regime
- **Future Work**: Extension to training scenarios, dynamic expert routing, larger models with thousands of experts