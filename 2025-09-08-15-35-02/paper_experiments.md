# Detailed Experiments: Large-Scale Cross-Node Expert Parallelism

## 1. Experimental Setup

### 1.1 Model Configuration

**Architecture Details**:
- **Model Type**: 4-layer Mixture-of-Experts (MoE) transformer
- **Experts per Layer**: 16 experts
- **Total Experts**: 64 experts (16 × 4 layers)
- **Expert Type**: Multi-Layer Perceptron (MLP)
- **Precision**: FP16 (16-bit floating point)

**Token Specifications**:
- **Batch Size**: 1024 sequences
- **Sequence Length**: 10,000 tokens per sequence
- **Total Tokens per Batch**: 10,240,000 tokens
- **Token Dimension**: 8192 (embedding/hidden size)

**Multi-Head Attention (MHA)**:
- **Number of Heads**: 16
- **Dimension per Head**: 512
- **Total MHA Dimension**: 8192 (16 × 512)

**MLP Expert Architecture**:
- **Input Size**: 8192
- **Hidden Size**: 32,768
- **Output Size**: 8192
- **Activation Function**: GELU
- **Parameters per Expert**: 8192×32768 + 32768×8192 = 536,870,912 parameters
- **Memory per Expert (FP16)**: 1,073,741,824 bytes = 1.024 GB

### 1.2 Hardware Configuration

**GPU Specifications**:
- **Type**: NVIDIA H100 Tensor Core GPUs
- **Memory per GPU**: 80 GB HBM3
- **Interconnect**: NVLink 4.0 + InfiniBand HDR
- **Node Configuration**: 8 GPUs per node (DGX H100)

## 2. Parallel Deployment Details

### 2.1 Baseline Deployment (TP=8, PP=2)

**Configuration Parameters**:
- **Total GPUs**: 16 H100 GPUs
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2
- **Expert Parallelism (EP)**: 2 (16 experts / 8 GPUs per stage)

**GPU Allocation**:
- **Pipeline Stage 1**: GPUs 0-7 (8 GPUs)
- **Pipeline Stage 2**: GPUs 8-15 (8 GPUs)
- **Per-GPU Load**: 4 experts + 1/8 tensor shard
- **Expert Distribution**: 4 experts per GPU (16 experts / 4 GPUs per expert group)

**Memory Usage per GPU**:
- **Tensor Shards**: 1/8 of model parameters
- **Experts**: 4 × 1GB = 4GB for expert parameters
- **Activations**: ~20GB for intermediate states
- **Total**: ~25GB per GPU

**Processing Flow**:
```
Input → Stage 1 (GPUs 0-7) → Stage 2 (GPUs 8-15) → Output
Each stage: 4 experts per GPU compete for compute
Sequential processing through pipeline stages
```

### 2.2 Proposed Cross-Node Expert Parallelism

**Configuration Parameters**:
- **Total GPUs**: 64 H100 GPUs
- **Tensor Parallelism (TP)**: 1 (per expert)
- **Pipeline Parallelism (PP)**: 4 (one stage per layer)
- **Expert Parallelism (EP)**: 64 (maximum possible)

**GPU Allocation per Layer**:
- **Layer 1**: GPUs 0-15 (16 experts)
- **Layer 2**: GPUs 16-31 (16 experts)
- **Layer 3**: GPUs 32-47 (16 experts)
- **Layer 4**: GPUs 48-63 (16 experts)
- **Per-GPU Load**: Exactly 1 expert

**Node Distribution**:
- **Node 1**: GPUs 0-7 (8 experts for Layer 1)
- **Node 2**: GPUs 8-15 (8 experts for Layer 1)
- **Node 3**: GPUs 16-23 (8 experts for Layer 2)
- **Node 4**: GPUs 24-31 (8 experts for Layer 2)
- **Node 5**: GPUs 32-39 (8 experts for Layer 3)
- **Node 6**: GPUs 40-47 (8 experts for Layer 3)
- **Node 7**: GPUs 48-55 (8 experts for Layer 4)
- **Node 8**: GPUs 56-63 (8 experts for Layer 4)

**Memory Usage per GPU**:
- **Expert Parameters**: 1GB (single expert)
- **Token Buffer**: 100MB (for received tokens)
- **Communication Buffer**: 256MB
- **Activations**: ~5GB (single expert processing)
- **Total**: ~6.5GB per GPU (significantly underutilized)

**Processing Flow**:
```
Input → All Layer 1 experts (parallel) → All Layer 2 experts (parallel) → All Layer 3 experts (parallel) → All Layer 4 experts (parallel) → Output
Each expert processes tokens independently on dedicated GPU
```

## 3. Performance Metrics

### 3.1 Throughput Comparison

| Method | GPUs Used | Configuration | TPS (Tokens/s) | Improvement |
|--------|-----------|---------------|----------------|-------------|
| Baseline | 16 | TP=8, PP=2, 4 experts/GPU | 120,000 | 1.0× |
| Proposed | 64 | EP=64, 1 expert/GPU | 450,000 | 3.75× |

**Detailed Breakdown**:
- **Baseline**: 120,000 tokens/s / 16 GPUs = 7,500 tokens/s per GPU
- **Proposed**: 450,000 tokens/s / 64 GPUs = 7,031 tokens/s per GPU
- **Efficiency**: 93.7% per-GPU efficiency maintained while scaling 4×

### 3.2 Latency Comparison

| Method | TPOT (ms) | Latency Reduction | Notes |
|--------|-----------|-------------------|-------|
| Baseline | 8.3 | - | Pipeline stalls and expert contention |
| Proposed | 2.2 | 3.8× faster | No contention, full parallelism |

**Latency Components**:
- **Baseline**: 5ms compute + 2.3ms pipeline stalls + 1ms communication
- **Proposed**: 1.8ms compute + 0.2ms routing + 0.2ms communication

### 3.3 Resource Utilization

**GPU Utilization**:
- **Baseline**: 70-80% (contention between 4 experts)
- **Proposed**: 95-98% (dedicated expert per GPU)

**Network Utilization**:
- **Baseline**: 20-30% (mostly intra-node communication)
- **Proposed**: 60-70% (cross-node token routing)

**Memory Utilization**:
- **Baseline**: 30-35GB used / 80GB available
- **Proposed**: 6.5GB used / 80GB available (intentional underutilization for compute focus)

## 4. Scaling Analysis

### 4.1 Strong Scaling Results

**Fixed Problem Size (10.24M tokens)**:
- 16 GPUs: 120,000 TPS
- 32 GPUs: 240,000 TPS (2.0× scaling)
- 64 GPUs: 450,000 TPS (3.75× scaling)
- **Scaling Efficiency**: 93.7% at 64 GPUs

### 4.2 Weak Scaling Results

**Fixed Tokens per GPU (160K tokens)**:
- 16 GPUs: 2.56M tokens → 120,000 TPS
- 32 GPUs: 5.12M tokens → 240,000 TPS
- 64 GPUs: 10.24M tokens → 480,000 TPS (ideal)
- **Measured**: 450,000 TPS (93.7% efficiency)

### 4.3 Communication Overhead Analysis

**Cross-Node Traffic**:
- **Token Volume**: 10.24M tokens × 8192 bytes = 83.9 GB per batch
- **Network Bandwidth**: 50 GB/s per GPU × 64 GPUs = 3.2 TB/s aggregate
- **Communication Time**: 83.9 GB / 3.2 TB/s = 26.2 ms (amortized across batch)

**Overlap Efficiency**:
- **Compute Time**: 1.8ms per expert per batch
- **Communication Time**: 0.4ms (overlapped with compute)
- **Idle Time**: <0.1ms per GPU

## 5. Bottleneck Analysis

### 5.1 Baseline Bottlenecks

**Primary Bottlenecks**:
1. **Intra-GPU Contention**: 4 experts sharing compute units
2. **Pipeline Stalls**: Sequential processing through stages
3. **Load Imbalance**: Uneven expert activation
4. **Memory Bandwidth**: Shared memory access patterns

### 5.2 Proposed Method Bottlenecks

**Primary Bottlenecks**:
1. **Network Bandwidth**: Cross-node token transfer
2. **Load Balancing**: Dynamic routing overhead
3. **Synchronization**: Coordination between layers
4. **Memory Underutilization**: Intentional for compute focus

### 5.3 Optimization Opportunities

**Future Improvements**:
- **Expert Caching**: Cache popular experts on multiple nodes
- **Dynamic Load Balancing**: Real-time routing adjustments
- **Topology Optimization**: Custom network topologies
- **Memory Compression**: Better utilize available memory

## 6. Validation and Reproducibility

### 6.1 Experimental Repeats

**Configuration**:
- **Runs per Configuration**: 5
- **Duration per Run**: 1000 batches
- **Warmup Batches**: 100
- **Measurement Window**: 900 batches

**Statistical Results**:
- **TPS Standard Deviation**: ±2.1%
- **TPOT Standard Deviation**: ±1.8%
- **Confidence Interval**: 95%

### 6.2 Hardware Consistency

**Validation Checks**:
- **GPU Model**: All H100-SXM5-80GB
- **Driver Version**: 535.54.03
- **CUDA Version**: 12.2
- **NCCL Version**: 2.18.3
- **Network**: InfiniBand HDR (200 Gbps)