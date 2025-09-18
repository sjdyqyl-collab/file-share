# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Model Type**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Architecture**: Multi-Layer Perceptron (MLP)
- **Precision**: FP16 (16-bit floating point)

### 1.2 Input Configuration
- **Batch Size**: 1024 sequences per batch
- **Sequence Length**: 10,000 tokens per sequence
- **Token Dimension**: 8,192 dimensions per token
- **Multi-Head Attention**: 
  - Number of heads: 16
  - Dimension per head: 512
  - Total MHA dimension: 16 × 512 = 8,192
- **MLP Hidden Size**: 32,768

### 1.3 Hardware Environment
- **GPU Type**: H100 GPUs
- **Setting**: Inference-only (no training)

## 2. Parallel Deployment Configurations

### 2.1 Baseline Configuration
- **Parallel Strategy**: TP=8, PP=2
- **GPUs Used**: 16 H100 GPUs
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of tensor-parallel shard for all layers
  - Pipeline stages: 2 stages total
  - GPUs per stage: 8 GPUs
  - Experts per GPU: 4 experts (colocated)
- **Processing Pattern**: Tokens flow sequentially through pipeline stages with shared compute resources

### 2.2 Proposed Method Configuration
- **Parallel Strategy**: Large EP (Expert Parallelism)
- **GPUs Used**: 64 H100 GPUs
- **Per-GPU Allocation**:
  - Each GPU hosts exactly **one expert**
  - Tensor parallelism: Optional TP=2 only if single expert exceeds GPU memory
  - Pipeline parallelism: Each MoE layer as micro-stage
- **Routing Strategy**: 
  - Dynamic token routing to GPU holding corresponding expert
  - Asynchronous token batch sending
  - Overlapped communication and computation

## 3. Performance Metrics

### 3.1 Primary Metrics
- **TPS (Tokens per Second)**: Overall throughput measurement
- **TPOT (Time per Output Token)**: Latency per individual token

### 3.2 Results Comparison

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### 3.3 Performance Improvements
- **Throughput Gain**: 450,000 ÷ 120,000 = **3.75× higher**
- **Latency Reduction**: 8.3 ÷ 2.2 = **3.8× lower**
- **GPU Utilization**: 4× more GPUs (16→64) yielding 3.75× throughput indicates near-linear scaling

## 4. Bottleneck Analysis

### 4.1 Baseline Limitations
- **Intra-GPU Contention**: 4 experts sharing GPU compute resources
- **Pipeline Stalls**: Sequential processing through pipeline stages
- **Resource Underutilization**: GPUs not fully utilized due to expert sharing

### 4.2 Proposed Method Advantages
- **Maximal Expert Parallelism**: All 64 experts compute simultaneously
- **No Resource Contention**: Single expert per GPU eliminates sharing
- **Overlapped Operations**: Communication hides latency through asynchronous routing
- **Near-Linear Scaling**: 4× GPU increase yields 3.75× performance gain

## 5. Scalability Validation

### 5.1 Large EP Regime Performance
- **EP Degree**: 16 (meeting large EP definition of EP ≥ 16)
- **Network Efficiency**: Modern HPC networking (NVLink, InfiniBand, NVSwitch) sustains required bandwidth
- **Compute Saturation**: One-expert-per-GPU ensures full GPU utilization

### 5.2 Communication Overhead Mitigation
- **Topology-Aware Placement**: Minimizes cross-node traffic
- **Token Batching**: Reduces network message count
- **Asynchronous Processing**: Overlaps communication with computation