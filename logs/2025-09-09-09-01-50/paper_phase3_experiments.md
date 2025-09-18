# Large-Scale Cross-Node Expert Parallelism - Detailed Experiments

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Experiments

### 1. Experimental Setup

#### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (16-bit floating point)
- **Activation Function**: GELU

#### 1.2 Input Configuration
- **Batch Size**: 1024 sequences per batch
- **Sequence Length**: 10000 tokens per sequence
- **Token Dimension**: 8192 dimensions per token
- **Total Tokens per Batch**: 10,240,000 tokens (1024 × 10000)

#### 1.3 Multi-Head Attention Parameters
- **Number of Heads**: 16 attention heads
- **Dimension per Head**: 512
- **Total MHA Dimension**: 8192 (16 × 512)

#### 1.4 MLP Expert Parameters
- **Hidden Size**: 32768 neurons
- **Input/Output Size**: 8192 (matches token dimension)
- **Activation**: GELU
- **Dropout**: 0.1 (during training, not used in inference)

#### 1.5 Hardware Configuration
- **GPU Type**: NVIDIA H100
- **GPU Memory**: 80GB HBM3 per GPU
- **Interconnect**: InfiniBand HDR (200 Gbps) + NVLink 4.0
- **CPU**: AMD EPYC 9654 (96 cores per node)
- **System Memory**: 2TB DDR5 per node

### 2. Parallel Deployment Configurations

#### 2.1 Baseline Configuration (TP=8, PP=2)
- **Total GPUs**: 16 H100 GPUs
- **Tensor Parallelism (TP)**: 8-way
- **Pipeline Parallelism (PP)**: 2 stages
- **Expert Parallelism (EP)**: 1 (experts colocated)
- **Data Parallelism (DP)**: 1

##### 2.1.1 GPU Allocation Details
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (16 total GPUs / 2 stages)
  - Experts colocated: 4 experts per GPU (64 total experts / 16 GPUs)
  - Memory per expert: ~20GB (shared with tensor shards)

##### 2.1.2 Processing Flow
1. **Stage 0 (Layers 1-2)**: 8 GPUs process first half of model
2. **Stage 1 (Layers 3-4)**: 8 GPUs process second half of model
3. **Expert Contention**: 4 experts per GPU share compute resources
4. **Communication**: 
   - Tensor parallel all-reduce within each stage
   - Pipeline send/recv between stages

#### 2.2 Proposed Cross-Node Expert Parallelism
- **Total GPUs**: 64 H100 GPUs
- **Expert Parallelism (EP)**: 64 (one expert per GPU)
- **Tensor Parallelism (TP)**: 1 (per expert, optional TP=2 if needed)
- **Pipeline Parallelism (PP)**: 4 stages (one per layer)
- **Data Parallelism (DP)**: 1

##### 2.2.1 GPU Allocation Details
- **Per-GPU Allocation**:
  - Each GPU hosts exactly one expert
  - Expert ID to GPU mapping: GPU i hosts expert i mod 64
  - Memory per expert: ~75GB (full expert parameters)
  - No intra-GPU expert contention

##### 2.2.2 Expert Distribution
- **Layer 1**: Experts 0-15 on GPUs 0-15
- **Layer 2**: Experts 16-31 on GPUs 16-31
- **Layer 3**: Experts 32-47 on GPUs 32-47
- **Layer 4**: Experts 48-63 on GPUs 48-63

##### 2.2.3 Routing Configuration
- **Dynamic Routing**: Tokens routed based on gating scores
- **Top-K Selection**: K=2 experts per token
- **Token Transfer**: Asynchronous send/recv between GPUs
- **Batching**: Group tokens by destination expert

### 3. Performance Results

#### 3.1 Throughput Comparison
| Method | GPUs Used | TPS (Tokens/Second) | Improvement |
|--------|-----------|---------------------|-------------|
| Baseline (TP=8, PP=2) | 16 | 120,000 | 1.0× |
| Proposed Cross-Node EP | 64 | 450,000 | 3.75× |

#### 3.2 Latency Comparison
| Method | GPUs Used | TPOT (ms) | Improvement |
|--------|-----------|-----------|-------------|
| Baseline (TP=8, PP=2) | 16 | 8.3 | 1.0× |
| Proposed Cross-Node EP | 64 | 2.2 | 3.77× |

#### 3.3 Detailed Metrics
- **Baseline Throughput Breakdown**:
  - GPU utilization: ~65% (due to expert contention)
  - Network utilization: ~40%
  - Memory utilization: ~85%

- **Proposed Method Throughput Breakdown**:
  - GPU utilization: ~95% (no expert contention)
  - Network utilization: ~75%
  - Memory utilization: ~94%

### 4. Scalability Analysis

#### 4.1 GPU Scaling
- **Linear Scaling Range**: 16-64 GPUs
- **Scaling Efficiency**: 94% (450K/480K theoretical max)
- **Communication Overhead**: 6% of total time

#### 4.2 Memory Usage
- **Per-GPU Memory**:
  - Model parameters: ~70GB
  - Activations: ~3GB
  - Communication buffers: ~2GB
  - Total: ~75GB (94% of 80GB)

#### 4.3 Network Traffic
- **Token Transfer Volume**: 10.2M tokens × 8192 dims × 2 bytes = 167GB per layer
- **Effective Bandwidth**: 450GB/s aggregate across all links
- **Link Utilization**: 75% of theoretical maximum

### 5. Bottleneck Analysis

#### 5.1 Baseline Bottlenecks
- **Primary**: Expert contention on GPUs (4 experts sharing compute)
- **Secondary**: Pipeline bubbles due to sequential processing
- **Tertiary**: Tensor parallelism overhead

#### 5.2 Proposed Method Bottlenecks
- **Primary**: Network bandwidth for token transfers
- **Secondary**: Load balancing across experts
- **Tertiary**: Synchronization overhead for final aggregation

### 6. Validation Details

#### 6.1 Correctness Verification
- **Numerical Accuracy**: FP16 precision maintained within 1e-3 relative error
- **Consistency**: All 64 GPUs produce deterministic results
- **Load Balance**: Maximum 5% imbalance across experts

#### 6.2 Performance Validation
- **Warmup**: 100 batches for stable measurements
- **Measurement**: Average of 1000 batches after warmup
- **Confidence Interval**: ±2% for throughput measurements