# Large-Scale Cross-Node Expert Parallelism - Experiments

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16
- **Expert Type**: MLP with GELU activation
- **Precision**: FP16 (16-bit floating point)

### 1.2 Input Configuration
- **Batch Size**: 1024 sequences
- **Sequence Length**: 10,000 tokens per sequence
- **Total Tokens per Batch**: 10,240,000 tokens
- **Token Dimension**: 8192

### 1.3 Multi-Head Attention Parameters
- **Number of Heads**: 16
- **Head Dimension**: 512
- **Total MHA Dimension**: 16 × 512 = 8192

### 1.4 MLP Expert Parameters
- **Hidden Size**: 32,768
- **Input/Output Size**: 8192 (matches token dimension)
- **Activation Function**: GELU

### 1.5 Hardware Configuration
- **GPU Type**: NVIDIA H100
- **Environment**: High-performance computing (HPC) cluster
- **Network**: InfiniBand/NVLink for cross-node communication

## 2. Parallel Deployment Details

### 2.1 Baseline Deployment (TP=8, PP=2)
- **Total GPUs**: 16 H100
- **Parallelism Configuration**:
  - Tensor Parallelism (TP): 8
  - Pipeline Parallelism (PP): 2
  - Expert Parallelism (EP): Not explicitly used
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs
  - 4 experts colocated per GPU
- **Processing Flow**:
  - Tokens flow sequentially through 2 pipeline stages
  - Multiple experts share GPU compute resources
  - Intra-GPU contention between experts

### 2.2 Proposed Cross-Node Expert Parallelism
- **Total GPUs**: 64 H100
- **Parallelism Configuration**:
  - Expert Parallelism (EP): 64 (16 experts × 4 layers)
  - Tensor Parallelism (TP): Optional TP=2 if needed for memory
  - Pipeline Parallelism (PP): Layer-wise micro-stages
- **Per-GPU Allocation**:
  - Each GPU hosts exactly one expert
  - 64 experts total across 4 layers (16 per layer)
  - No expert colocation on GPUs
- **Processing Flow**:
  - All 64 experts compute in parallel
  - Asynchronous token routing between experts
  - Communication overlapped with computation

## 3. Performance Metrics

### 3.1 Throughput Measurement
- **Metric**: Tokens per Second (TPS)
- **Baseline**: 120,000 TPS
- **Proposed**: 450,000 TPS
- **Improvement**: 3.75× increase

### 3.2 Latency Measurement
- **Metric**: Time per Output Token (TPOT)
- **Baseline**: 8.3 milliseconds
- **Proposed**: 2.2 milliseconds
- **Improvement**: 3.77× reduction

### 3.3 Resource Utilization
- **Baseline**: 16 GPUs, 4 experts/GPU → 64 expert instances
- **Proposed**: 64 GPUs, 1 expert/GPU → 64 expert instances
- **GPU Utilization**: 100% per expert in proposed vs shared in baseline

## 4. Results Summary

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) | Expert Distribution |
|--------|-----------|-------------------|----------------|-----------|-------------------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 | 4 experts/GPU |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 | 1 expert/GPU |

## 5. Scalability Analysis

### 5.1 Scaling Characteristics
- **Linear Scaling**: Near-linear throughput increase with GPU count
- **Communication Overhead**: Amortized across large token batches
- **Network Requirements**: High-bandwidth InfiniBand sustains all-to-all communication

### 5.2 Bottleneck Analysis
- **Baseline**: Intra-GPU contention, pipeline stalls
- **Proposed**: Network bandwidth, synchronization overhead
- **Break-even Point**: EP ≥ 16 shows clear advantages

## 6. Validation Details

### 6.1 Measurement Methodology
- **Warmup**: 100 batches for system stabilization
- **Measurement**: Average over 1000 batches
- **Metrics Collection**: CUDA events for precise timing

### 6.2 Reproducibility
- **Random Seed**: Fixed for consistent routing
- **Data**: Synthetic tokens with realistic distribution
- **Environment**: Isolated HPC nodes to prevent interference

## 7. Discussion of Results

### 7.1 Performance Gains
- **Primary Factor**: Elimination of intra-GPU expert contention
- **Secondary Factor**: Overlapped communication and computation
- **Tertiary Factor**: Better load balancing across nodes

### 7.2 Resource Trade-offs
- **GPU Count**: 4× increase (16 → 64 GPUs)
- **Throughput**: 3.75× increase (linear scaling efficiency: 94%)
- **Cost Efficiency**: Higher absolute performance, lower performance/GPU

### 7.3 Practical Implications
- **Use Case**: High-throughput inference with abundant GPU resources
- **Limitation**: Requires EP ≥ 16 for benefits
- **Future Work**: Extension to training scenarios with gradient synchronization