# Experiments Extraction - Large-Scale Cross-Node Expert Parallelism

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (half precision)
- **Total parameters**: Not explicitly stated, but can be calculated from dimensions

### 1.2 Input Configuration
- **Batch size**: 1024 sequences per batch
- **Sequence length**: 10,000 tokens per sequence
- **Token dimension**: 8192 dimensions per token
- **Total tokens per batch**: 10,240,000 tokens (1024 × 10,000)

### 1.3 Multi-Head Attention (MHA) Details
- **Number of heads**: 16 attention heads
- **Dimension per head**: 512
- **Total MHA dimension**: 16 × 512 = 8192 (matches token dimension)

### 1.4 MLP Expert Details
- **Hidden size**: 32,768 (4× token dimension, standard transformer ratio)
- **Activation function**: Not specified (typically GELU or ReLU in MoE models)
- **Expert structure**: Standard feed-forward network with expansion ratio 4

### 1.5 Hardware Configuration
- **GPU type**: NVIDIA H100
- **Total GPUs used**: 16 H100 GPUs
- **Environment**: High-performance computing (HPC) cluster
- **Network**: High-bandwidth interconnects (NVLink/InfiniBand)

## 2. Parallel Deployment Configurations

### 2.1 Baseline Configuration (TP=8, PP=2)
- **Parallelism degrees**:
  - Tensor Parallelism (TP): 8
  - Pipeline Parallelism (PP): 2
  - Expert Parallelism (EP): Not explicitly stated (experts colocated)
  - Data Parallelism (DP): 1 (single replica)

- **GPU allocation**:
  - Total GPUs: 16
  - Pipeline stages: 2 stages
  - GPUs per stage: 8 GPUs
  - Tensor parallelism within each stage: 8-way

- **Expert placement**:
  - 8 experts per layer per GPU (colocated)
  - Total experts per layer: 16
  - Experts shared among multiple GPUs via tensor parallelism
  - Each GPU holds 1/8 tensor-parallel shard for all layers

### 2.2 Proposed Cross-Node Expert Parallelism
- **Parallelism degrees**:
  - Expert Parallelism (EP): 16 (maximum possible with 16 GPUs)
  - Tensor Parallelism (TP): 1 (experts not further partitioned)
  - Pipeline Parallelism (PP): 1 (single pipeline stage)
  - Data Parallelism (DP): 1 (single replica)

- **GPU allocation**:
  - Total GPUs: 16
  - One GPU per expert per layer
  - Each GPU hosts exactly one expert per layer
  - 16 experts per layer × 4 layers = 64 expert instances total

- **Expert placement**:
  - Each GPU: 4 experts (one per layer)
  - Expert distribution: 16 GPUs × 1 expert per layer = 16 experts per layer
  - No expert colocation on same GPU

## 3. Performance Metrics

### 3.1 Throughput Measurements
- **Metric**: Tokens per Second (TPS)
- **Baseline (TP=8, PP=2)**: 120,000 tokens/second
- **Proposed (EP=16)**: 450,000 tokens/second
- **Improvement**: 3.75× increase in throughput

### 3.2 Latency Measurements
- **Metric**: Time per Output Token (TPOT)
- **Baseline (TP=8, PP=2)**: 8.3 milliseconds
- **Proposed (EP=16)**: 2.2 milliseconds
- **Improvement**: 3.77× reduction in latency

### 3.3 Efficiency Analysis
- **Token processing rate**: 450,000 tokens/second ÷ 16 GPUs = 28,125 tokens/second/GPU
- **Utilization improvement**: From 7,500 tokens/second/GPU (baseline) to 28,125 tokens/second/GPU
- **GPU efficiency gain**: 3.75× better GPU utilization

## 4. Detailed Performance Comparison

| Configuration | GPUs Used | Expert Placement | TPS | TPOT | GPU Utilization |
|---------------|-----------|------------------|-----|------|-----------------|
| Baseline | 16 | 8 experts/GPU + TP shard | 120,000 | 8.3ms | 7,500 tokens/s/GPU |
| Proposed | 16 | 1 expert/GPU | 450,000 | 2.2ms | 28,125 tokens/s/GPU |
| Improvement | - | - | 3.75× | 3.77× | 3.75× |

## 5. Bottleneck Analysis

### 5.1 Baseline Bottlenecks
- **Intra-GPU contention**: 8 experts sharing GPU compute resources
- **Pipeline stalls**: Sequential processing through 2 pipeline stages
- **Tensor parallelism overhead**: All-reduce operations across 8 GPUs
- **Memory bandwidth**: Multiple experts competing for GPU memory bandwidth

### 5.2 Proposed Solution Benefits
- **No intra-GPU contention**: Single expert per GPU
- **Maximal parallelism**: All 16 experts compute simultaneously
- **Reduced synchronization**: No tensor parallelism within experts
- **Better memory locality**: Single expert per GPU improves cache efficiency

## 6. Scalability Validation

### 6.1 Linear Scaling Verification
- **Theoretical maximum**: 16× single-expert performance
- **Achieved scaling**: 3.75× over baseline (which had suboptimal placement)
- **Communication overhead**: Minimal due to high-bandwidth interconnects
- **Load balancing**: Effective dynamic routing prevents stragglers

### 6.2 Network Requirements
- **Inter-node bandwidth**: High-bandwidth NVLink/InfiniBand sufficient
- **Latency tolerance**: Asynchronous communication masks latency
- **Topology optimization**: Expert placement considers network topology
- **Congestion management**: Token batching reduces network messages

## 7. Experimental Validation Details

### 7.1 Workload Characteristics
- **Compute intensity**: High (large MLP experts with 32K hidden size)
- **Memory intensity**: High (8192 token dimension, 10K sequence length)
- **Communication pattern**: All-to-all (tokens to experts)
- **Load distribution**: Dynamic based on input tokens

### 7.2 Measurement Methodology
- **Warmup**: Sufficient iterations to reach steady state
- **Averaging**: Multiple runs averaged for stable measurements
- **Metrics collection**: Hardware performance counters
- **End-to-end measurement**: Full inference pipeline timing

## 8. Reproducibility Information

### 8.1 Model Parameters
- **Total expert parameters**: 16 experts × 4 layers × (8192 × 32768 + 32768 × 8192) = 16 × 4 × 2 × 8192 × 32768 ≈ 34.4B parameters
- **Attention parameters**: 4 layers × (8192 × 8192 × 3 for QKV + 8192 × 8192 for output) ≈ 1.3B parameters
- **Total model size**: ~35.7B parameters in FP16 = ~71.4 GB

### 8.2 Configuration Files
- **Baseline config**: TP=8, PP=2, EP=implicit
- **Proposed config**: TP=1, PP=1, EP=16
- **Batch size**: Fixed at 1024 sequences for fair comparison
- **Precision**: FP16 throughout for consistency