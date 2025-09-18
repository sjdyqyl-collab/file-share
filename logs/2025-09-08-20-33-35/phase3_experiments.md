# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Type**: MLP (Multi-Layer Perceptron)
- **Total Experts**: 64 experts (4 layers × 16 experts/layer)

### 1.2 Precision and Dimensions
- **Precision**: FP16 (Half precision)
- **Token Dimension**: 8192
- **MHA (Multi-Head Attention)**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total MHA dimension: 16 × 512 = 8192
- **MLP Hidden Size**: 32768

### 1.3 Batch Configuration
- **Batch Size**: 1024 sequences per batch
- **Sequence Length**: 10000 tokens per sequence
- **Total Tokens per Batch**: 1024 × 10000 = 10,240,000 tokens

### 1.4 Hardware Environment
- **GPU Type**: H100 GPUs
- **Deployment**: Inference-only setting

### 1.5 Evaluation Metrics
- **TPS (Tokens per Second)**: Throughput measurement
- **TPOT (Time per Output Token)**: Latency measurement per token

## 2. Parallel Deployment Details

### 2.1 Baseline Deployment (TP=8, PP=2)
- **GPUs Used**: 16 H100 GPUs
- **Parallel Configuration**:
  - Tensor Parallelism (TP): 8-way
  - Pipeline Parallelism (PP): 2 stages
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (16 GPUs ÷ 2 stages = 8 GPUs/stage)
  - Experts colocated on GPUs: 4 experts per GPU
- **Processing Flow**:
  - Tokens flow sequentially through pipeline stages
  - Multiple experts per GPU share compute resources

### 2.2 Proposed Cross-Node Expert Parallelism
- **GPUs Used**: 64 H100 GPUs
- **Per-GPU Allocation**:
  - Each GPU hosts exactly **one expert**
  - Total experts: 64 (16 experts/layer × 4 layers = 64 experts)
  - Tensor Parallelism: Optional TP=2 if single expert's FFN cannot fit on one GPU
  - Pipeline Parallelism: Each MoE layer is a micro-stage
- **Routing Mechanism**:
  - Input tokens dynamically routed to GPU holding corresponding expert
  - Token batches asynchronously sent to minimize idle time
  - Communication of tokens overlapped with computation

## 3. Experimental Results

### 3.1 Performance Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 H100 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 H100 | 1 expert per GPU | 450,000 | 2.2 |

### 3.2 Performance Improvements
- **Throughput Gain**: 450,000 ÷ 120,000 = **3.75× higher TPS**
- **Latency Reduction**: 8.3 ÷ 2.2 = **3.8× lower TPOT**
- **GPU Utilization**: All 64 experts per layer compute in parallel

### 3.3 Detailed Analysis
- **Baseline Limitations**:
  - GPUs shared among multiple experts causing intra-GPU contention
  - Pipeline stalls due to sequential processing
  - Limited expert-level parallelism

- **Proposed Method Advantages**:
  - Dedicated one expert per GPU enables maximal expert-level parallelism
  - Asynchronous token routing ensures minimal waiting across nodes
  - Near-linear scaling with 64 GPUs in large EP regime (EP ≥ 16)

## 4. Scalability Validation
- **Large EP Regime**: EP=16 (16 experts per layer) achieved
- **Resource Utilization**: All 64 GPUs fully utilized for expert computation
- **Communication Overhead**: Successfully mitigated through overlapping computation and communication
- **Network Efficiency**: Topology-aware placement and token batching prevent network bottlenecks

## 5. Experimental Conclusion
The experiments validate that the proposed cross-node expert parallelism method achieves significant performance improvements by:
1. Maximizing expert-level parallelism through one-expert-per-GPU deployment
2. Effectively managing communication overhead through asynchronous routing and overlap
3. Demonstrating near-linear scaling in the large EP regime (EP ≥ 16)
4. Achieving 3.75× higher throughput and 3.8× lower latency compared to traditional approaches