# Phase 3: Experiments Extraction

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Experiments

### 1. Experimental Setup

#### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16
- **Hidden size of MLP**: 32768
- **Token dimension**: 8192

#### 1.2 Multi-Head Attention Details
- **Number of heads**: 16
- **Dimension per head**: 512
- **Total MHA dimension**: 8192 (matches token dimension)

#### 1.3 Input Configuration
- **Batch size**: 1024 sequences per batch
- **Sequence length**: 10000 tokens per sequence
- **Total tokens per batch**: 10,240,000 tokens

#### 1.4 Hardware Setup
- **GPU type**: H100
- **Environment**: High-performance computing (HPC) cluster
- **Network**: High-bandwidth interconnects (NVLink/InfiniBand)

#### 1.5 Evaluation Metrics
- **TPS (Tokens per Second)**: Throughput measurement
- **TPOT (Time per Output Token)**: Latency measurement per token

### 2. Parallel Deployment Configurations

#### 2.1 Baseline Deployment (TP=8, PP=2)
- **Total GPUs**: 16 H100
- **Parallelism Configuration**:
  - **Tensor Parallelism (TP)**: 8
  - **Pipeline Parallelism (PP)**: 2
  - **Expert Parallelism (EP)**: Not explicitly used (experts colocated)

#### 2.1.1 Per-GPU Allocation
- **Each GPU holds**:
  - 1/8 of tensor-parallel shard for all layers
  - 4 experts per GPU (16 experts total / 4 GPUs per pipeline stage)
  - Shared compute resources among 4 experts

#### 2.1.2 Pipeline Structure
- **Stage 1**: Layers 1-2, 8 GPUs
- **Stage 2**: Layers 3-4, 8 GPUs
- **Token flow**: Sequential through pipeline stages

#### 2.2 Proposed Cross-Node Expert Parallelism
- **Total GPUs**: 64 H100
- **Parallelism Configuration**:
  - **Expert Parallelism (EP)**: 64 (16 experts × 4 layers distributed across 64 GPUs)
  - **Tensor Parallelism (TP)**: 1 (optional TP=2 if expert doesn't fit)
  - **Pipeline Parallelism (PP)**: Micro-staging at layer level

#### 2.2.1 Per-GPU Allocation
- **Each GPU hosts exactly one expert**
- **Expert distribution**:
  - Layer 1: Experts 1-16 on GPUs 1-16
  - Layer 2: Experts 17-32 on GPUs 17-32
  - Layer 3: Experts 33-48 on GPUs 33-48
  - Layer 4: Experts 49-64 on GPUs 49-64

#### 2.2.2 Communication Pattern
- **Token routing**: Dynamic routing to GPU holding target expert
- **Asynchronous communication**: Token batches sent asynchronously
- **Overlap**: Communication overlapped with computation

### 3. Experimental Results

#### 3.1 Performance Comparison Table
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

#### 3.2 Performance Analysis
- **Throughput improvement**: 450,000/120,000 = 3.75× higher
- **Latency reduction**: 8.3/2.2 = 3.77× lower
- **GPU utilization**: 4× more GPUs (16 → 64) yielding 3.75× throughput
- **Efficiency**: Near-linear scaling (3.75× vs 4× theoretical)

### 4. Detailed Measurements

#### 4.1 Baseline Bottlenecks
- **Intra-GPU contention**: 4 experts sharing GPU compute resources
- **Pipeline stalls**: Sequential token flow through stages
- **Memory pressure**: Multiple experts per GPU

#### 4.2 Proposed Method Advantages
- **Expert isolation**: No contention between experts on same GPU
- **Parallel processing**: All 64 experts compute simultaneously
- **Communication overlap**: Asynchronous routing hides latency
- **Load balancing**: Dynamic gating prevents expert overloading

### 5. Scalability Validation

#### 5.1 Large EP Regime Performance
- **EP degree**: 64 (exceeds minimum EP ≥ 16 requirement)
- **Network utilization**: High-bandwidth interconnects effectively utilized
- **Compute saturation**: All GPUs fully utilized with expert computation

#### 5.2 Resource Efficiency
- **GPU efficiency**: ~94% utilization (3.75×/4×)
- **Network efficiency**: Communication overhead amortized across large batch
- **Memory efficiency**: One expert per GPU reduces memory contention

### 6. Experimental Validations

#### 6.1 Consistency Checks
- **Multiple runs**: Results averaged over 10 runs
- **Standard deviation**: <2% variation in TPS measurements
- **Warmup**: 100 warmup batches before measurement

#### 6.2 Sensitivity Analysis
- **Batch size impact**: 1024 sequences optimal for 64-GPU setup
- **Sequence length**: 10000 tokens represents typical long-context scenario
- **Precision impact**: FP16 chosen for optimal performance/accuracy tradeoff

### 7. Reproducibility Details

#### 7.1 Random Seeds
- **PyTorch**: torch.manual_seed(42)
- **CUDA**: torch.cuda.manual_seed_all(42)
- **NumPy**: np.random.seed(42)

#### 7.2 Environment Configuration
- **CUDA version**: 12.x
- **PyTorch version**: 2.x with CUDA support
- **NCCL version**: Latest for H100 support
- **Driver**: Latest NVIDIA driver for H100