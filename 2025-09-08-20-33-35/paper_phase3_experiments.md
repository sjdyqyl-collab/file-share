# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 (baseline) or 64 (proposed)
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (half precision)
- **Sequence parameters**:
  - Batch size: 1024 sequences
  - Sequence length: 10000 tokens per sequence
  - Token dimension: 8192
- **Multi-Head Attention (MHA)**:
  - Number of heads: 16
  - Dimension per head: 512
- **MLP specifications**:
  - Hidden size: 32768

### 1.2 Environment
- **Hardware**: H100 GPUs
- **Setting**: Inference-only (no training)
- **Metrics**:
  - TPS (Tokens per Second): throughput measurement
  - TPOT (Time per Output Token): latency per token measurement

## 2. Parallel Deployment Details

### 2.1 Baseline Deployment (TP=8, PP=2)
- **GPUs Used**: 16 H100
- **Parallelism Configuration**:
  - Tensor Parallelism (TP): 8
  - Pipeline Parallelism (PP): 2
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage (2 stages total) spans 8 GPUs
  - Experts are colocated on GPUs: typically 4 experts per GPU
- **Processing Flow**: Tokens flow sequentially through pipeline stages, multiple experts per GPU share compute resources

### 2.2 Proposed Cross-Node Expert Parallelism
- **GPUs Used**: 64 H100 (one GPU per expert per layer)
- **Per-GPU Allocation**:
  - Each GPU hosts exactly one expert
  - Tensor parallelism applied only if single expert's FFN cannot fit on one GPU (optional TP=2)
  - Pipeline parallelism: each MoE layer is a micro-stage; communication of tokens is overlapped with computation
- **Routing Mechanism**:
  - Input tokens dynamically routed to the GPU holding the corresponding expert
  - Token batches asynchronously sent to ensure minimal idle time
- **Expert Distribution**: All 64 experts per layer compute in parallel, maximizing throughput and minimizing token latency

## 3. Results

### 3.1 Performance Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### 3.2 Performance Analysis
- **Throughput Improvement**: ~3.75× higher (450,000 vs 120,000 TPS)
- **Latency Reduction**: ~3.8× lower (2.2ms vs 8.3ms TPOT)
- **Resource Utilization**: 
  - Baseline: GPUs shared among multiple experts, causing intra-GPU contention and pipeline stalls
  - Proposed: One expert per GPU enables maximal expert-level parallelism
- **Scaling Behavior**: Near-linear scaling with 64 GPUs in large EP regime (EP ≥ 16)

## 4. Discussion

### 4.1 Key Enablers
- **Dedicated Expert per GPU**: Allows full utilization of GPU compute and memory without contention
- **Asynchronous Token Routing**: Ensures minimal waiting even across nodes through overlapping communication and computation
- **Large EP Regime**: With 64 GPUs (unlimited H100s), system scales near-linearly when EP ≥ 16

### 4.2 Communication Optimization
- **Topology-aware placement**: Minimizes network congestion
- **Token batching**: Reduces network messages
- **Dynamic load balancing**: Prevents expert overloading and stragglers

### 4.3 Scalability Implications
- **Compute vs Communication Trade-off**: Method successfully shifts bottleneck from compute contention to communication, which is effectively managed through modern HPC networking
- **Memory Efficiency**: One-expert-per-GPU allows handling larger individual experts
- **Future Scaling**: Provides blueprint for even larger deployments with thousands of experts