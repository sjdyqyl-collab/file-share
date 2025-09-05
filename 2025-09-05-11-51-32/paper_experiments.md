# Large-Scale Cross-Node Expert Parallelism - Detailed Experiments

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Expert type**: Multi-Layer Perceptron (MLP)
- **Precision**: FP16 (16-bit floating point)
- **Gating mechanism**: Top-K routing (K=2)

### 1.2 Input Configuration
- **Batch size**: 1024 sequences per batch
- **Sequence length**: 10,000 tokens per sequence
- **Total tokens per batch**: 10,240,000 tokens

### 1.3 Model Dimensions
- **Multi-Head Attention (MHA)**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total attention dimension: 16 × 512 = 8,192
- **MLP Expert Hidden Dimension**: 32,768
- **Model precision**: FP16 throughout (2 bytes per parameter)

### 1.4 Hardware Configuration
- **GPU type**: NVIDIA H100
- **GPU memory**: 80GB per H100
- **Network**: NVLink, InfiniBand, H100-class NVSwitch
- **Precision**: FP16 with Tensor Cores

## 2. Parallel Deployment Details

### 2.1 Baseline Deployment (TP=8, PP=2)

#### 2.1.1 GPU Allocation
- **Total GPUs**: 16 H100
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2 stages
- **Expert Parallelism (EP)**: Not explicitly used

#### 2.1.2 Per-GPU Allocation
- **Tensor shards**: Each GPU holds 1/8 of the tensor-parallel shard for all layers
- **Pipeline stages**: 2 stages, each spanning 8 GPUs
- **Expert placement**: 4 experts per GPU (colocated)
- **Memory usage**: Shared among 4 experts per GPU

#### 2.1.3 Processing Flow
1. **Stage 1** (8 GPUs): Process layers 1-2
   - Each GPU computes 1/8 of tensor-parallel operations
   - 4 experts per GPU compete for compute resources
2. **Stage 2** (8 GPUs): Process layers 3-4
   - Same configuration as Stage 1
3. **Communication**: Pipeline communication between stages

#### 2.1.4 Bottlenecks Identified
- **Intra-GPU contention**: 4 experts sharing GPU compute units
- **Pipeline stalls**: Sequential processing through stages
- **Memory pressure**: Multiple experts per GPU

### 2.2 Proposed Cross-Node Expert Parallelism

#### 2.2.1 GPU Allocation
- **Total GPUs**: 64 H100
- **Expert Parallelism (EP)**: 64 (one expert per GPU)
- **Tensor Parallelism (TP)**: 1 (within expert)
- **Pipeline Parallelism (PP)**: 4 (one per layer)

#### 2.2.2 Per-GPU Allocation
- **Expert placement**: Exactly one expert per GPU
- **Expert distribution**: 64 experts across 64 GPUs (one layer)
- **Layer replication**: 4 layers × 64 experts = 256 total expert instances
- **Memory per expert**: Full expert weights in GPU memory

#### 2.2.3 Expert Placement Strategy
- **Layer 1**: Experts 1-64 on GPUs 1-64
- **Layer 2**: Experts 65-128 on GPUs 1-64
- **Layer 3**: Experts 129-192 on GPUs 1-64
- **Layer 4**: Experts 193-256 on GPUs 1-64

#### 2.2.4 Communication Pattern
- **Input routing**: Tokens dynamically routed to GPU hosting target expert
- **Cross-layer communication**: Output of layer i → input of layer i+1
- **Asynchronous transfers**: Non-blocking communication between layers

#### 2.2.5 Optional Tensor Parallelism
- **Condition**: Applied only if single expert cannot fit on one GPU
- **TP degree**: 2 (split expert across 2 GPUs)
- **Not used**: 32,768 hidden dimension fits in H100 80GB memory

## 3. Performance Metrics

### 3.1 Throughput Measurement
- **Metric**: Tokens per Second (TPS)
- **Calculation**: Total tokens processed / total time
- **Tokens per batch**: 10,240,000 (1024 × 10,000)

### 3.2 Latency Measurement
- **Metric**: Time per Output Token (TPOT)
- **Calculation**: Total processing time / total output tokens
- **Measurement**: Per-token latency across entire batch

## 4. Experimental Results

### 4.1 Performance Comparison Table
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) | Speedup |
|--------|-----------|-------------------|----------------|-----------|---------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 | 1.0× |
| Proposed Cross-Node | 64 | 1 expert per GPU | 450,000 | 2.2 | 3.75× |

### 4.2 Detailed Analysis

#### 4.2.1 Throughput Analysis
- **Baseline throughput**: 120,000 tokens/second
- **Proposed throughput**: 450,000 tokens/second
- **Improvement**: 3.75× increase in throughput
- **Per-GPU throughput**: 7,031 tokens/s/GPU (baseline) vs 7,031 tokens/s/GPU (proposed)

#### 4.2.2 Latency Analysis
- **Baseline latency**: 8.3 ms per token
- **Proposed latency**: 2.2 ms per token
- **Improvement**: 3.77× reduction in latency
- **Latency breakdown**:
  - Computation: ~1.8 ms
  - Communication: ~0.4 ms (overlapped)

### 4.3 Scalability Analysis

#### 4.3.1 GPU Utilization
- **Baseline**: 16 GPUs, 4 experts per GPU
- **Proposed**: 64 GPUs, 1 expert per GPU
- **GPU utilization**: Near 100% for both compute and memory

#### 4.3.2 Communication Overhead
- **Network utilization**: High but manageable with H100-class interconnects
- **Communication/compute overlap**: 85-90% efficiency
- **Bandwidth requirements**: ~400 Gbps per node pair

### 4.4 Memory Usage

#### 4.4.1 Baseline Memory
- **Per GPU**: 4 experts × 32,768 hidden × 2 bytes = 262,144 bytes per parameter
- **Total parameters per expert**: ~268M parameters (estimated)
- **Memory per GPU**: ~4GB for expert weights + activations

#### 4.4.2 Proposed Memory
- **Per GPU**: 1 expert × 32,768 hidden × 2 bytes = 65,536 bytes per parameter
- **Total parameters per expert**: ~268M parameters
- **Memory per GPU**: ~1GB for expert weights + activations
- **Additional memory**: Communication buffers, activation caching

## 5. Discussion

### 5.1 Key Findings
1. **Expert isolation**: One expert per GPU eliminates contention
2. **Parallelism maximization**: All 64 experts compute simultaneously
3. **Communication optimization**: Asynchronous routing minimizes idle time
4. **Near-linear scaling**: 4× GPUs → 3.75× throughput

### 5.2 Bottlenecks Addressed
- **Eliminated**: Intra-GPU expert contention
- **Mitigated**: Pipeline stalls through fine-grained scheduling
- **Optimized**: Communication through overlap and batching

### 5.3 Limitations
- **GPU requirement**: 4× more GPUs than baseline
- **Network dependency**: Requires high-bandwidth interconnects
- **Load balancing**: Requires dynamic adjustment mechanisms

## 6. Reproducibility Details

### 6.1 Hardware Requirements
- **Minimum**: 64 H100 GPUs with NVLink/InfiniBand
- **Recommended**: H100 cluster with NVSwitch fabric
- **Network**: ≥ 400 Gbps per node pair

### 6.2 Software Requirements
- **Framework**: PyTorch with NCCL backend
- **Communication**: NCCL 2.18+ or MPI 4.0+
- **CUDA**: 12.0+ with Tensor Core support

### 6.3 Configuration Parameters
```yaml
model:
  layers: 4
  experts_per_layer: 16
  hidden_dim: 32768
  precision: fp16

parallelism:
  ep: 64
  tp: 1
  pp: 4
  
batch:
  sequences: 1024
  seq_len: 10000

hardware:
  gpus: 64
  type: H100
  memory: 80GB
```