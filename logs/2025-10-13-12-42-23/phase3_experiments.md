# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (half precision)

### 1.2 Input Configuration
- **Batch Size**: 1024 sequences per batch
- **Sequence Length**: 10,000 tokens per sequence
- **Token Dimension**: 8,192 dimensions per token
- **Multi-Head Attention**: 16 heads, 512 dimensions per head
- **MLP Hidden Size**: 32,768 hidden units

### 1.3 Hardware Environment
- **GPU Type**: H100 GPUs
- **Environment**: High-performance computing (HPC) cluster
- **Precision**: FP16 for all computations

### 1.4 Evaluation Metrics
- **TPS (Tokens per Second)**: Measures overall throughput
- **TPOT (Time per Output Token)**: Measures latency per token

## 2. Parallel Deployment Configurations

### 2.1 Baseline Configuration (TP=8, PP=2)
- **Total GPUs**: 16 H100 GPUs
- **Tensor Parallelism (TP)**: 8-way partitioning
- **Pipeline Parallelism (PP)**: 2 stages
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (16 GPUs / 2 stages = 8 GPUs per stage)
  - Experts are colocated on GPUs: typically 8 experts per layer per GPU
- **Processing Flow**: Tokens flow sequentially through pipeline stages
- **Resource Sharing**: Multiple experts per GPU share compute resources

### 2.2 Proposed Cross-Node Expert Parallelism
- **Total GPUs**: 16 H100 GPUs
- **Expert Parallelism (EP)**: 16 (one GPU per expert per layer)
- **Per-GPU Allocation**:
  - Each GPU hosts exactly one expert per layer
  - Total: 4 layers × 16 experts/layer = 64 expert instances across 16 GPUs
- **Expert Distribution**: 16 experts per layer distributed across 16 GPUs (1 expert per GPU)
- **Routing Strategy**: 
  - Input tokens dynamically routed to GPU holding corresponding expert
  - Token batches asynchronously sent to minimize idle time
- **Parallelism Focus**: Maximize expert-level parallelism (EP ≥ 16)

## 3. Performance Results

### 3.1 Throughput Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) | Performance Ratio |
|--------|-----------|-------------------|----------------|-----------|-------------------|
| Baseline (TP=8, PP=2) | 16 | 8 experts each layer + TP shard per GPU | 120,000 | 8.3 | 1.0× |
| Proposed Cross-Node Expert Parallelism | 16 | 1 expert each layer per GPU | 450,000 | 2.2 | 3.75× |

### 3.2 Performance Analysis
- **Throughput Improvement**: 3.75× increase (450,000 vs 120,000 TPS)
- **Latency Reduction**: 3.8× decrease (2.2ms vs 8.3ms TPOT)
- **Resource Utilization**: Full GPU utilization with dedicated expert per GPU
- **Contention Elimination**: No intra-GPU contention between experts

## 4. Detailed Configuration Parameters

### 4.1 Model Dimensions
```
Layer Count: 4
Experts per Layer: 16
Token Dimension: 8192
MHA Heads: 16
MHA Head Dimension: 512
MLP Hidden Size: 32768
Precision: FP16
```

### 4.2 Batch Configuration
```
Sequences per Batch: 1024
Tokens per Sequence: 10000
Total Tokens per Batch: 10,240,000
```

### 4.3 Parallelism Parameters
```
Baseline:
  Tensor Parallelism (TP): 8
  Pipeline Parallelism (PP): 2
  Expert Parallelism (EP): Not explicitly defined (experts colocated)
  GPUs: 16

Proposed:
  Expert Parallelism (EP): 16
  Tensor Parallelism (TP): 1 (within expert)
  Pipeline Parallelism (PP): 1 (within layer)
  GPUs: 16
```

## 5. Implementation Notes

### 5.1 Baseline Limitations
- **Intra-GPU Contention**: Multiple experts share GPU resources
- **Pipeline Stalls**: Sequential processing through pipeline stages
- **Resource Underutilization**: GPU compute units not fully utilized

### 5.2 Proposed Method Advantages
- **Maximal Expert Parallelism**: All 16 experts per layer compute in parallel
- **Dedicated Resources**: Each expert has exclusive GPU access
- **Asynchronous Operations**: Minimal waiting through overlapping communication
- **Near-linear Scaling**: Demonstrated scalability in large EP regime (EP ≥ 16)

## 6. Experimental Environment Details

### 6.1 Hardware Specifications
- **GPU Model**: NVIDIA H100
- **Interconnect**: High-bandwidth NVLink/InfiniBand
- **Cluster Size**: 16 GPUs minimum for EP=16
- **Memory**: Sufficient for FP16 precision with 32K MLP hidden size

### 6.2 Software Stack
- **Precision**: FP16 throughout
- **Communication**: NCCL for cross-node communication
- **Scheduling**: Asynchronous token routing implementation
- **Load Balancing**: Dynamic gating probability adjustment

## 7. Scalability Validation

### 7.1 Large EP Regime Confirmation
- **EP=16**: Successfully demonstrated with 16 GPUs
- **Network Overhead**: Mitigated through topology-aware routing
- **Compute Saturation**: Achieved full GPU utilization
- **Communication Overlap**: Effective overlapping of compute and communication

### 7.2 Performance Scaling
- **Linear Scaling**: Near-linear throughput increase with GPU count
- **Latency Reduction**: Consistent latency improvement with increased parallelism
- **Resource Efficiency**: Optimal GPU utilization in HPC environment