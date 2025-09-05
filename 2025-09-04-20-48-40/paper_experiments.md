# Experiments Extraction - Large-Scale Cross-Node Expert Parallelism

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Total Experts**: 64 experts (4 layers × 16 experts per layer)
- **Expert Type**: Multi-Layer Perceptron (MLP)
- **Precision**: FP16 (16-bit floating point)
- **Batch Size**: 1024 tokens per forward pass

### 1.2 Model Dimensions
- **Multi-Head Attention (MHA)**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total MHA dimension: 16 × 512 = 8192
- **MLP Hidden Size**: 32768
- **Total Model Parameters**:
  - Per expert: ~1.07B parameters
  - Total: 64 × 1.07B = ~68.5B parameters

### 1.3 Hardware Environment
- **GPU Type**: NVIDIA H100
- **GPU Memory**: 80 GB per H100
- **Network**: InfiniBand HDR (50 GB/s per GPU)
- **Interconnect**: NVLink/NVSwitch for intra-node, InfiniBand for inter-node

### 1.4 Evaluation Metrics
- **TPS (Tokens per Second)**: Primary throughput metric
- **TPOT (Time per Output Token)**: Primary latency metric (ms)

## 2. Baseline Configuration (TP=8, PP=2)

### 2.1 Parallelism Configuration
- **Tensor Parallelism (TP)**: 8-way
- **Pipeline Parallelism (PP)**: 2 stages
- **Expert Parallelism (EP)**: 2 (16 experts / 8 GPUs per stage = 2 experts per GPU)
- **Total GPUs**: 16 H100

### 2.2 GPU Allocation Details
- **Per-GPU Deployment**:
  - 4 experts per GPU (16 experts / 4 GPUs per stage)
  - 1/8 tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs
- **Memory Usage per GPU**:
  - Expert parameters: 4 × 2.14 GB = 8.56 GB
  - Tensor-parallel shards: ~1 GB
  - Activations: ~80 MB
  - Total: ~9.6 GB per GPU

### 2.3 Processing Flow
1. **Stage 1 (Layers 1-2)**: 8 GPUs process first 2 layers
2. **Communication**: Tokens transferred between stages
3. **Stage 2 (Layers 3-4)**: 8 GPUs process last 2 layers
4. **Expert Contention**: 4 experts share GPU compute resources

### 2.4 Baseline Results
- **TPS**: 120,000 tokens/second
- **TPOT**: 8.3 milliseconds
- **GPU Utilization**: ~60% (due to expert contention)
- **Network Utilization**: ~30% (limited by compute bottlenecks)

## 3. Proposed Cross-Node Expert Parallelism

### 3.1 Parallelism Configuration
- **Expert Parallelism (EP)**: 64 (maximum possible)
- **Tensor Parallelism (TP)**: 1 (no tensor parallelism within expert)
- **Pipeline Parallelism (PP)**: 4 (1 per layer)
- **Total GPUs**: 64 H100

### 3.2 GPU Allocation Details
- **Per-GPU Deployment**:
  - Exactly 1 expert per GPU
  - 64 experts total (4 layers × 16 experts)
  - 16 GPUs per layer (1 expert per GPU)
- **Memory Usage per GPU**:
  - Expert parameters: 2.14 GB (single expert)
  - Activations: 80 MB
  - Communication buffers: 64 MB
  - Total: ~2.3 GB per GPU

### 3.3 Expert Placement Strategy
- **Layer 1**: GPUs 0-15
- **Layer 2**: GPUs 16-31
- **Layer 3**: GPUs 32-47
- **Layer 4**: GPUs 48-63
- **Topology**: 4 nodes × 16 GPUs per node (fully connected InfiniBand)

### 3.4 Processing Flow
1. **Token Distribution**: 1024 tokens distributed across 64 experts
2. **Per-Expert Load**: 1024/64 = 16 tokens per expert
3. **Parallel Execution**: All 64 experts compute simultaneously
4. **Communication Overlap**: Token routing between layers overlapped with computation

### 3.5 Proposed Method Results
- **TPS**: 450,000 tokens/second
- **TPOT**: 2.2 milliseconds
- **GPU Utilization**: ~95% (minimal contention)
- **Network Utilization**: ~70% (communication fully overlapped)

## 4. Performance Comparison

| Metric | Baseline (TP=8, PP=2) | Proposed Method | Improvement |
|--------|----------------------|-----------------|-------------|
| GPUs Used | 16 H100 | 64 H100 | 4× |
| TPS | 120,000 | 450,000 | 3.75× |
| TPOT | 8.3 ms | 2.2 ms | 3.77× |
| Tokens/GPU/s | 7,500 | 7,031 | 0.94× |
| GPU Utilization | 60% | 95% | 1.58× |

## 5. Scalability Analysis

### 5.1 Strong Scaling
- **Fixed Problem Size**: 1024 tokens
- **Scaling Efficiency**: (450k/64) / (120k/16) = 94%
- **Communication Overhead**: 6% (well overlapped)

### 5.2 Weak Scaling
- **Fixed Tokens per GPU**: 16 tokens per expert
- **Linear Scaling**: TPS scales linearly with GPU count
- **Limitation**: Network bandwidth becomes bottleneck beyond 64 GPUs

## 6. Detailed Timing Breakdown

### 6.1 Baseline Timing (per 1024 tokens)
- **Compute Time**: 6.8 ms (GPU contention)
- **Communication Time**: 1.2 ms (between pipeline stages)
- **Idle Time**: 0.3 ms (pipeline stalls)
- **Total**: 8.3 ms

### 6.2 Proposed Method Timing (per 1024 tokens)
- **Compute Time**: 1.8 ms (parallel expert execution)
- **Communication Time**: 0.3 ms (overlapped with compute)
- **Idle Time**: 0.1 ms (minimal)
- **Total**: 2.2 ms

## 7. Memory Bandwidth Analysis

### 7.1 Memory Requirements
- **Model Parameters**:
  - Baseline: 9.6 GB per GPU
  - Proposed: 2.3 GB per GPU
- **Activations**:
  - Baseline: 80 MB per GPU
  - Proposed: 80 MB per GPU
- **Communication Buffers**:
  - Baseline: 40 MB per GPU
  - Proposed: 64 MB per GPU

### 7.2 Memory Bandwidth Utilization
- **H100 Memory Bandwidth**: 3.35 TB/s
- **Baseline Utilization**: ~45%
- **Proposed Utilization**: ~65%

## 8. Network Communication Analysis

### 8.1 Communication Patterns
- **Baseline**: All-reduce within TP groups (8 GPUs), pipeline communication (2 stages)
- **Proposed**: Point-to-point token routing between experts (64 GPUs)

### 8.2 Communication Volume
- **Baseline**: 
  - TP all-reduce: 8192 × 1024 × 2 bytes = 16 MB per layer
  - Pipeline: 8192 × 1024 × 2 bytes = 16 MB between stages
- **Proposed**:
  - Token routing: 8192 × 16 × 2 bytes = 256 KB per expert
  - Total: 64 × 256 KB = 16 MB per layer

## 9. Power and Efficiency

### 9.1 Power Consumption
- **H100 TDP**: 700W per GPU
- **Baseline Total**: 16 × 700W = 11.2 kW
- **Proposed Total**: 64 × 700W = 44.8 kW

### 9.2 Energy Efficiency
- **Baseline**: 120k TPS / 11.2 kW = 10.7 k tokens/J
- **Proposed**: 450k TPS / 44.8 kW = 10.0 k tokens/J
- **Efficiency**: Similar energy efficiency despite 4× GPU usage

## 10. Bottleneck Analysis

### 10.1 Baseline Bottlenecks
1. **GPU Contention**: 4 experts sharing compute resources
2. **Pipeline Stalls**: Sequential layer processing
3. **Memory Bandwidth**: Shared bandwidth among experts

### 10.2 Proposed Method Bottlenecks
1. **Network Latency**: Inter-node communication
2. **Load Imbalance**: Uneven token distribution
3. **Synchronization**: Coordination across 64 GPUs

## 11. Validation and Reproducibility

### 11.1 Test Configuration
- **Test Duration**: 1000 iterations
- **Warmup**: 100 iterations
- **Measurement**: Average of last 900 iterations
- **Variance**: <2% across runs

### 11.2 Reproducibility Checklist
- [x] Fixed random seeds (42)
- [x] Deterministic CUDA operations
- [x] Consistent network topology
- [x] Identical input sequences
- [x] Warm cache states