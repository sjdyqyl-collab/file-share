# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Model Type**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Total Experts**: 64 experts (16 × 4 layers)
- **Expert Architecture**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (16-bit floating point)
- **Attention Configuration**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total attention dimension: 16 × 512 = 8192
- **MLP Hidden Size**: 32768

### 1.2 Input Configuration
- **Batch Size**: 1024 sequences per batch
- **Sequence Length**: 10000 tokens per sequence
- **Total Tokens per Batch**: 1024 × 10000 = 10,240,000 tokens

### 1.3 Hardware Configuration
- **GPU Type**: H100 GPUs
- **Environment**: Inference-only setting
- **Interconnect**: High-performance network (NVLink, InfiniBand, NVSwitch)

### 1.4 Evaluation Metrics
- **TPS (Tokens per Second)**: Throughput measurement
- **TPOT (Time per Output Token)**: Latency measurement per token

## 2. Parallel Deployment Configurations

### 2.1 Baseline Deployment (TP=8, PP=2)
- **Total GPUs**: 16 H100 GPUs
- **Parallel Strategy**:
  - **Tensor Parallelism (TP)**: 8-way
  - **Pipeline Parallelism (PP)**: 2 stages
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs
  - **Experts per GPU**: 4 experts colocated on each GPU
- **Processing Flow**:
  - Tokens flow sequentially through 2 pipeline stages
  - Multiple experts per GPU share compute resources
  - Intra-GPU contention between experts

### 2.2 Proposed Cross-Node Expert Parallelism
- **Total GPUs**: 64 H100 GPUs
- **Parallel Strategy**:
  - **Expert Parallelism (EP)**: 64 (16 experts × 4 layers)
  - **Tensor Parallelism (TP)**: Optional 2-way (only if expert doesn't fit)
  - **Pipeline Parallelism**: Each MoE layer as micro-stage
- **Per-GPU Allocation**:
  - **One expert per GPU**: Exactly one expert per GPU
  - **Memory usage**: Each expert has dedicated GPU memory
  - **Compute utilization**: Full GPU dedicated to single expert
- **Routing Strategy**:
  - Dynamic token routing to GPU holding corresponding expert
  - Asynchronous token batch transfer
  - Overlapped communication with computation

## 3. Experimental Results

### 3.1 Performance Comparison Table
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### 3.2 Performance Improvements
- **Throughput Improvement**: 450,000 ÷ 120,000 = **3.75× higher**
- **Latency Reduction**: 8.3 ÷ 2.2 = **3.77× lower**
- **GPU Utilization**: 64 vs 16 GPUs = **4× more GPUs used**
- **Efficiency Gain**: Despite 4× more GPUs, achieved 3.75× throughput

### 3.3 Scalability Analysis
- **Linear Scaling**: Near-linear scaling achieved in large EP regime (EP ≥ 16)
- **Resource Utilization**: Full GPU compute utilization per expert
- **Communication Overhead**: Mitigated through asynchronous routing and overlap

## 4. Experimental Environment Details

### 4.1 Network Configuration
- **Interconnect Type**: High-bandwidth, low-latency HPC networking
- **Bandwidth**: Sufficient to sustain large EP communication patterns
- **Topology**: Optimized for cross-node expert distribution

### 4.2 Memory Constraints
- **Per-GPU Memory**: Sufficient for single expert + attention components
- **Memory Balancing**: Even distribution across all 64 GPUs
- **Optional TP=2**: Used only when single expert exceeds GPU memory

### 4.3 Load Balancing Results
- **Expert Utilization**: Balanced across all 64 experts
- **Token Distribution**: Dynamic gating prevents expert overload
- **Straggler Prevention**: No significant performance degradation from slow experts

## 5. Discussion Points

### 5.1 Bottleneck Analysis
- **Baseline**: Intra-GPU contention between 4 experts per GPU
- **Proposed**: Network communication (effectively managed)

### 5.2 Resource Requirements
- **Baseline**: 16 H100 GPUs with shared expert resources
- **Proposed**: 64 H100 GPUs with dedicated expert per GPU

### 5.3 Cost-Benefit Analysis
- **GPU Cost**: 4× increase in GPU count
- **Performance Gain**: 3.75× throughput improvement
- **Efficiency**: 93.75% scaling efficiency (3.75/4.0)

## 6. Experimental Validations

### 6.1 Reproducibility
- **Fixed Configuration**: All parameters explicitly defined
- **Deterministic Routing**: Consistent expert placement
- **Stable Results**: Multiple runs averaged for final metrics

### 6.2 Edge Cases Tested
- **Memory Limits**: Single expert fitting within GPU memory
- **Network Congestion**: High EP scenarios with maximum token transfer
- **Load Imbalance**: Dynamic gating handling skewed token distributions