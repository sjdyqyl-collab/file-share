# Experiments: Large-Scale Cross-Node Expert Parallelism

## 1. Experimental Setup

### 1.1 Model Configuration
- **Model Type**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Architecture**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (16-bit floating point)

### 1.2 Input Configuration
- **Batch Size**: 1024 sequences per batch
- **Sequence Length**: 10,000 tokens per sequence
- **Token Dimension**: 8,192
- **Multi-Head Attention**: 16 heads, 512 dimension per head
- **MLP Hidden Size**: 32,768

### 1.3 Hardware Setup
- **GPUs**: H100 GPUs
- **Total GPUs**: 16
- **Setting**: Inference-only

### 1.4 Evaluation Metrics
- **TPS (Tokens per Second)**: Measures throughput
- **TPOT (Time per Output Token)**: Measures latency per token

## 2. Parallel Deployment Configurations

### 2.1 Baseline Deployment (TP=8, PP=2)
- **GPUs Used**: 16 H100
- **Tensor Parallelism (TP)**: 8-way
- **Pipeline Parallelism (PP)**: 2 stages
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs
  - 8 experts per GPU per layer (colocated)
- **Processing**: Sequential pipeline flow with shared compute resources

### 2.2 Proposed Cross-Node Expert Parallelism
- **GPUs Used**: 16 H100
- **Expert Parallelism (EP)**: 16-way
- **Per-GPU Allocation**:
  - Each GPU hosts exactly one expert per layer
- **Routing**:
  - Dynamic token routing to GPU holding corresponding expert
  - Asynchronous token batch sending
  - Minimal idle time through overlap

## 3. Results

### 3.1 Performance Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 8 experts each layer + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 16 | 1 expert each layer per GPU | 450,000 | 2.2 |

### 3.2 Performance Improvements
- **Throughput Improvement**: 3.75× higher (450,000 vs 120,000 TPS)
- **Latency Improvement**: 3.8× lower (2.2ms vs 8.3ms TPOT)
- **Expert Utilization**: 100% (16 experts computing in parallel)
- **Resource Efficiency**: Full GPU utilization without contention

## 4. Discussion

### 4.1 Key Advantages
- **No Intra-GPU Contention**: One expert per GPU eliminates resource sharing
- **Full Parallelism**: All 16 experts compute simultaneously
- **Asynchronous Routing**: Minimal waiting time across nodes
- **Near-Linear Scaling**: Effective for EP ≥ 16 regime

### 4.2 Baseline Limitations
- **Shared Resources**: Multiple experts per GPU cause contention
- **Pipeline Stalls**: Sequential processing creates bottlenecks
- **Underutilization**: GPUs not fully utilized due to sharing

### 4.3 Scalability Insights
- **Large EP Regime**: EP ≥ 16 enables optimal performance
- **Network Efficiency**: Communication costs amortized across tokens
- **Topology Awareness**: Expert placement minimizes network hotspots
- **Load Balancing**: Dynamic routing prevents expert overload

## 5. Experimental Validation

### 5.1 Throughput Validation
- **Measured TPS**: 450,000 tokens/second
- **Theoretical Scaling**: Near-linear with expert count
- **Network Impact**: Minimal due to asynchronous routing

### 5.2 Latency Validation
- **Measured TPOT**: 2.2ms per token
- **Latency Reduction**: 3.8× improvement over baseline
- **Consistency**: Stable across different batch sizes

### 5.3 Resource Utilization
- **GPU Utilization**: 100% for all 16 GPUs
- **Memory Efficiency**: Optimal per-GPU memory usage
- **Network Bandwidth**: Effectively managed through batching