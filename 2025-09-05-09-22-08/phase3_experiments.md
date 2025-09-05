# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Type**: Each expert is a Multi-Layer Perceptron (MLP)
- **Precision**: FP16 (16-bit floating point)
- **Batch Size**: 1024 tokens per forward pass

### 1.2 Attention Configuration
- **Multi-Head Attention (MHA)**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total attention dimension: 16 × 512 = 8192

### 1.3 Expert Configuration
- **MLP Hidden Size**: 32,768
- **Activation Function**: Not explicitly stated (typically GELU or ReLU in MoE models)
- **Expert Capacity**: Not explicitly stated (typically 1.0-1.25x tokens per expert)

### 1.4 Hardware Environment
- **GPU Type**: H100 GPUs
- **Interconnect**: High-bandwidth fabric (NVLink, InfiniBand, or NVSwitch)
- **Setting**: Inference-only (no training evaluation)

## 2. Evaluation Metrics

### 2.1 Throughput Metric
- **TPS (Tokens per Second)**: Total tokens processed per second
- **Measurement**: Aggregate across all GPUs in the system

### 2.2 Latency Metric
- **TPOT (Time per Output Token)**: Average time to process each output token
- **Measurement**: End-to-end latency from input to output

## 3. Baseline Configuration (TP=8, PP=2)

### 3.1 Resource Allocation
- **GPUs Used**: 16 H100 GPUs
- **Tensor Parallelism (TP)**: 8-way splitting
- **Pipeline Parallelism (PP)**: 2 stages

### 3.2 Per-GPU Deployment
- **Each GPU contains**:
  - 1/8 of tensor-parallel shard for all layers
  - 4 experts per GPU (16 experts ÷ 4 GPUs per pipeline stage)
  - Pipeline stage spans 8 GPUs (16 GPUs ÷ 2 stages)

### 3.3 Processing Flow
- **Sequential**: Tokens flow through pipeline stages
- **Contention**: Multiple experts share GPU compute resources
- **Communication**: TP all-reduce within stages, PP send/recv between stages

## 4. Proposed Method Configuration

### 4.1 Resource Allocation
- **GPUs Used**: 64 H100 GPUs
- **Expert Parallelism (EP)**: 64-way (16 experts × 4 layers distributed)
- **One Expert per GPU**: Each GPU hosts exactly one expert

### 4.2 Parallelism Strategy
- **Tensor Parallelism (TP)**: Optional TP=2 if single expert cannot fit on GPU
- **Pipeline Parallelism**: Each MoE layer as a micro-stage
- **Expert Distribution**: 64 experts per layer (4 layers × 16 experts)

### 4.3 Routing Implementation
- **Dynamic Routing**: Input tokens routed to GPU holding corresponding expert
- **Asynchronous Transfer**: Token batches sent asynchronously
- **Overlap**: Communication overlapped with computation

## 5. Performance Results

### 5.1 Quantitative Comparison
| Method | GPUs | Per-GPU Deployment | TPS | TPOT |
|--------|------|-------------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 ms |
| Proposed Cross-Node EP | 64 | 1 expert per GPU | 450,000 | 2.2 ms |

### 5.2 Performance Improvements
- **Throughput Gain**: 450,000 ÷ 120,000 = 3.75× improvement
- **Latency Reduction**: 8.3 ÷ 2.2 = 3.77× improvement (approximately 3.8× as stated)
- **GPU Efficiency**: 4× more GPUs (64 vs 16) yielding 3.75× throughput

### 5.3 Scalability Analysis
- **Linear Scaling Factor**: 3.75× throughput with 4× GPUs indicates 93.75% scaling efficiency
- **Communication Overhead**: Minimal impact due to overlapping strategy
- **Resource Utilization**: Full GPU compute utilization with single expert per GPU

## 6. Discussion of Results

### 6.1 Throughput Analysis
- **Baseline Bottleneck**: Intra-GPU contention from multiple experts
- **Proposed Advantage**: Dedicated GPU resources per expert eliminate contention
- **Network Impact**: Communication overhead mitigated by asynchronous routing

### 6.2 Latency Analysis
- **Pipeline Stalls**: Baseline suffers from pipeline bubbles
- **Expert Isolation**: Proposed method enables true expert-level parallelism
- **Token Routing**: Dynamic routing minimizes waiting time

### 6.3 Resource Scaling
- **GPU Proportionality**: 4× GPUs providing 3.75× performance
- **Memory Bandwidth**: Each expert has dedicated GPU memory bandwidth
- **Compute Units**: Full GPU compute resources available per expert

## 7. Experimental Limitations

### 7.1 Scope Constraints
- **Inference Only**: No training performance evaluation
- **Fixed Model Size**: Only 4-layer, 16-expert configuration tested
- **Single Precision**: Only FP16 evaluated

### 7.2 Scale Limitations
- **Maximum GPUs**: 64 GPUs maximum tested
- **Expert Count**: Limited to 16 experts per layer
- **Batch Size**: Fixed at 1024 tokens per forward pass

## 8. Reproducibility Requirements

### 8.1 Hardware Requirements
- **Minimum**: 16 H100 GPUs for baseline
- **Optimal**: 64 H100 GPUs for proposed method
- **Interconnect**: High-bandwidth fabric essential

### 8.2 Software Requirements
- **Framework**: MoE implementation supporting EP, TP, PP
- **Communication**: NCCL or MPI for cross-node communication
- **CUDA**: Stream support for asynchronous operations