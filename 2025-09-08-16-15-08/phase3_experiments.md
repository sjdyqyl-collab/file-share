# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Model Type**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Architecture**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (16-bit floating point)
- **Batch Size**: 1024 sequences per batch
- **Sequence Length**: 10,000 tokens per sequence
- **Token Dimension**: 8192 dimensions per token
- **Multi-Head Attention (MHA)**: 
  - Number of heads: 16
  - Dimension per head: 512
- **MLP Hidden Size**: 32,768

### 1.2 Environment
- **Hardware**: H100 GPUs
- **Setting**: Inference-only (no training)
- **Metrics**:
  - TPS (Tokens per Second): Throughput measurement
  - TPOT (Time per Output Token): Latency per token

## 2. Parallel Deployment Details

### 2.1 Baseline Deployment (TP=8, PP=2)
- **GPUs Used**: 16 H100 GPUs
- **Parallel Configuration**:
  - Tensor Parallelism (TP): 8
  - Pipeline Parallelism (PP): 2 stages
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (2 stages total)
  - Experts are colocated on GPUs: typically 4 experts per GPU
- **Processing Flow**: Tokens flow sequentially through pipeline stages, multiple experts per GPU share compute resources

### 2.2 Proposed Cross-Node Expert Parallelism
- **GPUs Used**: 64 H100 GPUs
- **Per-GPU Allocation**:
  - Each GPU hosts **exactly one expert**
  - Tensor parallelism applied only if single expert's FFN cannot fit on one GPU (optional TP=2)
  - Pipeline parallelism: each MoE layer is a micro-stage
- **Routing Strategy**:
  - Input tokens dynamically routed to GPU holding corresponding expert
  - Token batches asynchronously sent to ensure minimal idle time
  - Communication of tokens overlapped with computation

## 3. Results

### 3.1 Performance Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### 3.2 Performance Improvements
- **Throughput**: ~3.75× higher (450,000 vs 120,000 TPS)
- **Latency**: ~3.8× lower (2.2ms vs 8.3ms TPOT)
- **Resource Utilization**: Full utilization of all 64 GPUs for expert-level parallelism

## 4. Analysis

### 4.1 Baseline Limitations
- **Intra-GPU Contention**: Multiple experts sharing GPU compute resources
- **Pipeline Stalls**: Sequential processing through pipeline stages
- **Resource Sharing**: 4 experts per GPU causing computational bottlenecks

### 4.2 Proposed Method Advantages
- **Maximal Expert Parallelism**: One expert per GPU eliminates contention
- **Concurrent Processing**: All 64 experts per layer compute in parallel
- **Asynchronous Operations**: Minimal waiting time even across nodes
- **Near-linear Scaling**: Achieved with 64 GPUs in large EP regime (EP ≥ 16)

## 5. Deployment Characteristics

### 5.1 Scalability
- **Large EP Regime**: EP = 16 (16 experts per layer × 4 layers = 64 total experts)
- **Network Efficiency**: Topology-aware routing and token batching
- **Communication Overlap**: Asynchronous token transfers with computation

### 5.2 Resource Requirements
- **GPU Count**: 64 H100 GPUs for full deployment
- **Memory**: Each GPU hosts single expert with dedicated resources
- **Network**: High-bandwidth interconnects (NVLink, InfiniBand) for cross-node communication

## 6. Experimental Validation
- **Setting**: Inference-only validation
- **Throughput**: 450,000 tokens/second achieved
- **Latency**: 2.2ms per token
- **Scalability**: Demonstrated near-linear scaling with increased GPU count
- **Efficiency**: Full GPU utilization without intra-GPU expert contention