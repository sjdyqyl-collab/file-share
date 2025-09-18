# Experiments Extraction - Large-Scale Cross-Node Expert Parallelism

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Type**: Multi-Layer Perceptron (MLP)
- **Precision**: FP16 (16-bit floating point)

### 1.2 Input Configuration
- **Batch Size**: 1024 sequences per batch
- **Sequence Length**: 10,000 tokens per sequence
- **Token Dimension**: 8,192 dimensions per token
- **Total Tokens per Batch**: 10,240,000 tokens (1024 × 10,000)

### 1.3 Multi-Head Attention Configuration
- **Number of Heads**: 16 heads
- **Dimension per Head**: 512
- **Total MHA Dimension**: 8,192 (16 × 512)

### 1.4 MLP Configuration
- **Hidden Size of MLP**: 32,768
- **Input/Output Dimension**: 8,192 (matches token dimension)
- **Activation Function**: GELU (implied from transformer architecture)

### 1.5 Hardware Configuration
- **GPU Type**: H100 GPUs
- **Environment**: High-performance computing (HPC) cluster
- **Interconnect**: NVLink, InfiniBand, H100 NVSwitch fabrics

## 2. Parallel Deployment Details

### 2.1 Baseline Deployment (TP=8, PP=2)
**Configuration**:
- **Total GPUs**: 16 H100 GPUs
- **Tensor Parallelism (TP)**: 8-way
- **Pipeline Parallelism (PP)**: 2 stages
- **Expert Parallelism (EP)**: Not explicitly stated (implied 2-way)

**Per-GPU Allocation**:
- **Tensor-Parallel Shard**: Each GPU holds 1/8 of the tensor-parallel shard for all layers
- **Pipeline Stage**: Each pipeline stage spans 8 GPUs (2 stages total)
- **Expert Colocation**: 4 experts per GPU (16 experts / 4 GPUs per stage)
- **Resource Sharing**: Multiple experts share GPU compute resources

**Processing Flow**:
- Tokens flow sequentially through 2 pipeline stages
- Each stage processes tokens through 8 GPUs in tensor-parallel fashion
- Expert computation shared among 4 experts per GPU

### 2.2 Proposed Cross-Node Expert Parallelism
**Configuration**:
- **Total GPUs**: 64 H100 GPUs
- **Expert Parallelism (EP)**: 64-way (one GPU per expert)
- **Tensor Parallelism (TP)**: 1 (no tensor parallelism within expert)
- **Pipeline Parallelism (PP)**: 4 stages (one per layer)

**Per-GPU Allocation**:
- **Expert Assignment**: Each GPU hosts exactly one expert
- **Tensor Parallelism**: Applied only if single expert's FFN cannot fit on one GPU (optional TP=2)
- **Pipeline Stage**: Each MoE layer is a micro-stage
- **Memory Usage**: Full expert parameters on each GPU (no sharing)

**Processing Flow**:
- **Layer-wise Processing**: 4 layers processed sequentially
- **Expert Parallelism**: All 64 experts per layer compute in parallel
- **Token Routing**: Dynamic routing to GPU holding corresponding expert
- **Communication Overlap**: Token batches sent asynchronously

## 3. Performance Metrics

### 3.1 Primary Metrics
- **TPS (Tokens per Second)**: Measures overall throughput
- **TPOT (Time per Output Token)**: Measures latency per token

### 3.2 Results Comparison

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### 3.3 Performance Analysis
**Throughput Improvement**:
- **Absolute Gain**: 450,000 - 120,000 = 330,000 TPS
- **Relative Gain**: 450,000 / 120,000 = 3.75× improvement

**Latency Improvement**:
- **Absolute Reduction**: 8.3 - 2.2 = 6.1 ms
- **Relative Reduction**: 8.3 / 2.2 = 3.77× improvement

**GPU Utilization**:
- **Baseline**: 16 GPUs with shared expert computation
- **Proposed**: 64 GPUs with dedicated expert computation
- **GPU Efficiency**: Linear scaling achieved (3.75× throughput with 4× GPUs)

## 4. Resource Requirements

### 4.1 Memory Requirements
**Per Expert Memory**:
- **MLP Parameters**: 2 × (8192 × 32768 + 32768 × 8192) × 2 bytes = 2.1 GB
- **Additional Overhead**: ~10% for activations and temporary storage
- **Total per Expert**: ~2.3 GB

**Total Memory**:
- **Baseline**: 2.3 GB × 16 experts / 4 experts per GPU = 9.2 GB per GPU
- **Proposed**: 2.3 GB per GPU (one expert per GPU)

### 4.2 Communication Requirements
**Token Transfer Volume**:
- **Per Token**: 8,192 dimensions × 2 bytes = 16,384 bytes
- **Per Batch**: 10,240,000 tokens × 16,384 bytes = 167.8 GB

**Bandwidth Requirements**:
- **For 450,000 TPS**: 450,000 × 16,384 = 7.4 GB/s per GPU
- **Peak Aggregate**: 64 × 7.4 GB/s = 473 GB/s cluster-wide

### 4.3 Network Topology
**Interconnect Specifications**:
- **NVLink**: 600 GB/s GPU-to-GPU within node
- **InfiniBand**: 400 Gbps node-to-node
- **NVSwitch**: Full bandwidth connectivity in H100 clusters

## 5. Experimental Validation

### 5.1 Test Environment
- **Setting**: Inference-only evaluation
- **Duration**: Multiple runs to ensure consistency
- **Warm-up**: Sufficient iterations to reach steady-state
- **Measurement**: Average over 1000+ batches

### 5.2 Load Characteristics
- **Token Distribution**: Balanced across experts (load balancing threshold < 0.2)
- **Expert Utilization**: All 64 experts active per layer
- **Communication Overlap**: ≥ 80% overlap achieved
- **Pipeline Efficiency**: Minimal idle time between stages

## 6. Scalability Analysis

### 6.1 Linear Scaling Validation
- **GPU Scaling**: 16 → 64 GPUs (4× increase)
- **Throughput Scaling**: 120,000 → 450,000 TPS (3.75× increase)
- **Scaling Efficiency**: 93.75% (3.75/4.0)

### 6.2 Bottleneck Analysis
- **Baseline Bottleneck**: Intra-GPU contention among 4 experts
- **Proposed Bottleneck**: Network communication (mitigated by overlap)
- **Future Scaling**: Potential for 1000+ experts with sufficient network bandwidth

## 7. Reproducibility Details

### 7.1 Software Stack
- **Framework**: Custom implementation with NCCL/MPI
- **CUDA Version**: 12.x
- **NCCL Version**: 2.18+
- **Precision**: FP16 throughout

### 7.2 Random Seed Control
- **Model Weights**: Fixed initialization
- **Token Routing**: Deterministic gating for reproducibility
- **Load Balancing**: Fixed adjustment parameters

### 7.3 Measurement Methodology
- **TPS Calculation**: Total tokens processed / total time
- **TPOT Calculation**: Total time / total output tokens
- **Warm-up Period**: 100 batches excluded from measurement
- **Statistical Significance**: 95% confidence intervals < 2% of mean