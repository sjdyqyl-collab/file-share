# Phase 3: Experiments Extraction - Large-Scale Cross-Node Expert Parallelism

## 1. Experimental Setup

### 1.1 Model Configuration
- **Model Type**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Architecture**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (half-precision floating point)
- **Batch Size**: 1024 tokens per forward pass
- **Multi-Head Attention (MHA)**:
  - Number of heads: 16
  - Dimension per head: 512
- **MLP Hidden Size**: 32,768

### 1.2 Evaluation Mode
- **Setting**: Inference-only (no training)
- **Hardware**: H100 GPUs
- **Metrics**:
  - TPS (Tokens per Second): Measures throughput
  - TPOT (Time per Output Token): Measures latency per token

## 2. Parallel Deployment Configurations

### 2.1 Baseline Configuration (TP=8, PP=2)
- **Total GPUs**: 16 H100
- **Parallelism Strategy**:
  - Tensor Parallelism (TP): 8-way
  - Pipeline Parallelism (PP): 2 stages
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (2 stages total)
  - Experts colocated on GPUs: 4 experts per GPU
- **Processing Flow**: Tokens flow sequentially through pipeline stages with shared compute resources

### 2.2 Proposed Cross-Node Expert Parallelism
- **Total GPUs**: 64 H100
- **Parallelism Strategy**:
  - Expert Parallelism (EP): 64-way (16 experts × 4 layers distributed)
  - Tensor Parallelism (TP): Optional TP=2 if single expert's FFN cannot fit on one GPU
  - Pipeline Parallelism: Each MoE layer as micro-stage
- **Per-GPU Allocation**:
  - Each GPU hosts exactly one expert
  - No expert colocation - complete isolation
- **Routing Strategy**:
  - Input tokens dynamically routed to GPU holding corresponding expert
  - Token batches asynchronously sent to minimize idle time
  - Communication overlapped with computation

## 3. Performance Results

### 3.1 Quantitative Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### 3.2 Performance Improvements
- **Throughput Gain**: 3.75× higher (450,000 vs 120,000 TPS)
- **Latency Reduction**: 3.8× lower (2.2ms vs 8.3ms TPOT)
- **Resource Utilization**: 4× more GPUs utilized (64 vs 16)
- **Expert Isolation**: Complete elimination of intra-GPU expert contention

## 4. Experimental Analysis

### 4.1 Bottleneck Analysis
- **Baseline Limitations**:
  - Intra-GPU contention from 4 experts sharing resources
  - Pipeline stalls between stages
  - Limited expert-level parallelism
- **Proposed Advantages**:
  - Maximal expert-level parallelism (64 experts computing simultaneously)
  - No resource contention within GPU
  - Near-linear scaling in large EP regime (EP ≥ 16)

### 4.2 Scalability Characteristics
- **Linear Scaling**: With 64 GPUs, system scales near-linearly
- **Communication Overhead**: Mitigated by asynchronous token routing
- **Network Requirements**: High-bandwidth interconnects essential (NVLink/InfiniBand)
- **Load Balancing**: Dynamic adjustment prevents expert overloading

## 5. Deployment Implications

### 5.1 Resource Requirements
- **Minimum Configuration**: 64 H100 GPUs for full expert isolation
- **Network Infrastructure**: High-bandwidth, low-latency interconnects
- **Memory**: Each GPU must accommodate one complete expert
- **Software**: NCCL/MPI for asynchronous communication

### 5.2 Practical Considerations
- **GPU Utilization**: 100% compute utilization per GPU for expert processing
- **Communication Pattern**: All-to-all token exchanges between layers
- **Fault Tolerance**: Expert replication strategy for E > G scenarios
- **Energy Efficiency**: Higher performance per watt due to reduced contention