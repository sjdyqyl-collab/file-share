# Experiments: Large-Scale Cross-Node Expert Parallelism for MoE Models

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Total Experts**: 64 experts (16 × 4 layers)
- **Expert Type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16
- **Token Dimension**: 8192
- **MHA Configuration**: 16 heads, 512 dimension per head
- **MLP Hidden Size**: 32768

### 1.2 Batch Configuration
- **Batch Size**: 1024 sequences
- **Sequence Length**: 10000 tokens per sequence
- **Total Tokens per Batch**: 10,240,000 tokens

### 1.3 Hardware Setup
- **GPU Type**: H100
- **Environment**: High-performance computing (HPC) cluster
- **Setting**: Inference-only

### 1.4 Evaluation Metrics
- **TPS (Tokens per Second)**: Measures overall throughput
- **TPOT (Time per Output Token)**: Measures latency per token

## 2. Parallel Deployment Details

### 2.1 Baseline Deployment (TP=8, PP=2)
- **GPUs Used**: 16 H100
- **Parallel Configuration**:
  - **Tensor Parallelism (TP)**: 8
  - **Pipeline Parallelism (PP)**: 2
  - **Expert Parallelism (EP)**: Not explicitly used (experts colocated)
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage (2 stages total) spans 8 GPUs
  - Experts are colocated on GPUs: typically 4 experts per GPU
- **Processing Flow**:
  - Tokens flow sequentially through the pipeline stages
  - Multiple experts per GPU share compute resources
  - Intra-GPU contention occurs due to multiple experts per GPU

### 2.2 Proposed Cross-Node Expert Parallelism
- **GPUs Used**: 64 H100
- **Parallel Configuration**:
  - **Expert Parallelism (EP)**: 64 (one expert per GPU)
  - **Tensor Parallelism (TP)**: Optional TP=2 if single expert cannot fit on one GPU
  - **Pipeline Parallelism (PP)**: Each MoE layer acts as a micro-stage
- **Per-GPU Allocation**:
  - Each GPU hosts exactly one expert
  - No intra-GPU expert contention
  - Full utilization of GPU compute for single expert
- **Routing Strategy**:
  - Input tokens dynamically routed to the GPU holding the corresponding expert
  - Token batches asynchronously sent to minimize idle time
  - Communication of tokens overlapped with computation
- **Expert Distribution**:
  - 64 experts per layer distributed across 64 GPUs
  - Each layer has dedicated 16 GPUs (16 experts × 4 layers = 64 GPUs total)

## 3. Experimental Results

### 3.1 Performance Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### 3.2 Performance Analysis
- **Throughput Improvement**: 3.75× higher (450,000 vs 120,000 TPS)
- **Latency Reduction**: 3.8× lower (2.2ms vs 8.3ms TPOT)
- **GPU Utilization**: 4× more GPUs used (64 vs 16)
- **Per-GPU Efficiency**: Each GPU dedicated to single expert eliminates contention

### 3.3 Scalability Characteristics
- **Linear Scaling**: Near-linear scaling achieved in large EP regime (EP ≥ 16)
- **Communication Overhead**: Effectively mitigated through asynchronous routing and compute-communication overlap
- **Resource Utilization**: All 64 GPUs fully utilized for expert computation

## 4. Discussion

### 4.1 Baseline Limitations
- **Intra-GPU Contention**: Multiple experts sharing GPU compute resources
- **Pipeline Stalls**: Sequential processing through pipeline stages
- **Resource Underutilization**: GPU compute not fully utilized due to expert sharing

### 4.2 Proposed Method Advantages
- **Maximal Expert Parallelism**: All 64 experts compute in parallel across 64 GPUs
- **No Intra-GPU Contention**: Each expert has dedicated GPU resources
- **Asynchronous Processing**: Minimal waiting time through overlapped communication and computation
- **Topology Awareness**: Expert placement considers network topology for optimal communication

### 4.3 Large EP Regime Benefits
- **Compute Saturation**: All GPUs fully utilized for expert computation
- **Communication Amortization**: Network communication costs spread across many tokens
- **Scalability**: System scales effectively with available GPU resources
- **HPC Environment Suitability**: Particularly effective in high-performance computing clusters

## 5. Experimental Validation

### 5.1 Throughput Validation
- **Measured TPS**: 450,000 tokens/second with 64 GPUs
- **Theoretical Scaling**: 4× GPU increase yielding 3.75× throughput improvement demonstrates near-linear scaling
- **Efficiency**: 93.75% scaling efficiency (3.75/4.0)

### 5.2 Latency Validation
- **Measured TPOT**: 2.2ms per token
- **Latency Reduction**: Consistent with throughput improvement, indicating efficient parallelization
- **End-to-end Performance**: Significant improvement in both throughput and latency metrics

## 6. Conclusion from Experiments

The experimental results validate the effectiveness of the proposed large-scale cross-node expert parallelism method. By distributing experts across GPUs with one expert per GPU and leveraging asynchronous communication, the system achieves significant performance improvements over traditional colocation strategies. The 3.75× throughput improvement and 3.8× latency reduction demonstrate the viability of the approach for large-scale MoE deployments in HPC environments.