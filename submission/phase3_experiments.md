# Phase 3: Experiments Extraction

## 1. Experimental Setup

### Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16
- **Batch size**: 1024 tokens per forward pass
- **Multi-head attention**: 16 heads, 512 dimensions per head
- **MLP hidden size**: 32,768

### Hardware Environment
- **GPUs**: H100 GPUs
- **Setting**: Inference-only

### Metrics
- **TPS (Tokens per Second)**: Throughput measurement
- **TPOT (Time per Output Token)**: Latency per token measurement

## 2. Parallel Deployment Details

### 2.1 Baseline Deployment (TP=8, PP=2)
- **GPUs Used**: 16 H100
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage (2 stages total) spans 8 GPUs
  - Experts are colocated on GPUs: typically 4 experts per GPU
- **Processing**: Tokens flow sequentially through pipeline stages, with multiple experts per GPU sharing compute resources

### 2.2 Proposed Cross-Node Expert Parallelism
- **GPUs Used**: 64 H100 (one GPU per expert per layer)
- **Per-GPU Allocation**:
  - Each GPU hosts exactly one expert
  - Tensor parallelism applied only if single expert's FFN cannot fit on one GPU (optional TP=2)
  - Pipeline parallelism: each MoE layer is a micro-stage; communication of tokens is overlapped with computation
- **Routing**:
  - Input tokens dynamically routed to the GPU holding the corresponding expert
  - Token batches asynchronously sent, ensuring minimal idle time

### Deployment Comparison
- **Total experts per layer**: 64 (4 layers × 16 experts)
- **Expert distribution**: All 64 experts per layer compute in parallel
- **Parallelism strategy**: Maximizes expert-level parallelism with minimal contention

## 3. Results

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### Performance Analysis
- **Throughput improvement**: 3.75× higher (450k vs 120k TPS)
- **Latency reduction**: 3.8× lower (2.2ms vs 8.3ms TPOT)
- **Resource utilization**: Baseline GPUs shared among multiple experts causing intra-GPU contention and pipeline stalls
- **Method advantage**: Dedicates one expert per GPU, enabling maximal expert-level parallelism

## 4. Discussion
- **One expert per GPU**: Allows full utilization of GPU compute and memory
- **Asynchronous token routing**: Ensures minimal waiting even across nodes
- **Scalability**: With 64 GPUs, system scales near-linearly in large EP regime (EP ≥ 16)
- **Network requirements**: Relies on high-performance interconnects (NVLink, InfiniBand, NVSwitch) to sustain bandwidth and low latency

## 5. Experimental Validations
- **Inference-only setting**: Demonstrates effectiveness for serving scenarios
- **FP16 precision**: Balances performance and memory efficiency
- **Large batch**: 1024 tokens effectively amortizes communication overhead
- **Cross-node communication**: Successfully overlaps with computation to achieve near-linear scaling