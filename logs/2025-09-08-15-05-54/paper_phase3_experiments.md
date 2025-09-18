# Phase 3: Experimental Results and Setup

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Total experts**: 64 (4 layers × 16 experts)
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16

### 1.2 Input Configuration
- **Batch size**: 1024 sequences per batch
- **Sequence length**: 10000 tokens per sequence
- **Total tokens per batch**: 10,240,000 tokens
- **Token dimension**: 8192

### 1.3 Attention Configuration
- **Multi-Head Attention (MHA)**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total MHA dimension: 16 × 512 = 8192

### 1.4 MLP Configuration
- **Hidden size of MLP**: 32768
- **Activation function**: Not specified (assumed GELU or ReLU based on standard practice)

### 1.5 Hardware Configuration
- **GPU type**: H100
- **Precision**: FP16
- **Setting**: Inference-only

## 2. Parallel Deployment Details

### 2.1 Baseline Deployment (TP=8, PP=2)
- **Total GPUs**: 16 H100
- **Parallelism strategy**:
  - Tensor Parallelism (TP): 8
  - Pipeline Parallelism (PP): 2 stages
- **Per-GPU allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (16 total GPUs ÷ 2 stages = 8 GPUs per stage)
  - Experts are colocated on GPUs: 4 experts per GPU (64 total experts ÷ 16 GPUs = 4 experts/GPU)
- **Processing flow**:
  - Tokens flow sequentially through pipeline stages
  - Multiple experts per GPU share compute resources
  - Intra-GPU contention occurs due to colocated experts

### 2.2 Proposed Cross-Node Expert Parallelism
- **Total GPUs**: 64 H100
- **Parallelism strategy**:
  - Expert Parallelism (EP): 64 (one expert per GPU)
  - Tensor Parallelism (TP): Optional TP=2 if single expert's FFN cannot fit on one GPU
  - Pipeline Parallelism: Each MoE layer as a micro-stage
- **Per-GPU allocation**:
  - Each GPU hosts exactly one expert
  - Total experts: 64 (4 layers × 16 experts) = 64 GPUs
  - No expert colocation - complete isolation
- **Routing mechanism**:
  - Input tokens dynamically routed to GPU holding corresponding expert
  - Token batches sent asynchronously to minimize idle time
  - Communication overlapped with computation

## 3. Performance Results

### 3.1 Throughput Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) | Improvement |
|--------|-----------|-------------------|----------------|-----------|-------------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 | 1× |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 | 3.75× |

### 3.2 Key Performance Metrics
- **Throughput improvement**: 450,000 ÷ 120,000 = 3.75× higher
- **Latency reduction**: 8.3 ÷ 2.2 = 3.77× lower (approximately 3.8× as stated)
- **GPU utilization**: 100% compute utilization per GPU (one expert per GPU)
- **Scalability**: Near-linear scaling achieved with 64 GPUs

### 3.3 Bottleneck Analysis
- **Baseline bottlenecks**:
  - Intra-GPU contention from 4 colocated experts
  - Pipeline stalls due to sequential processing
  - Shared compute resources limiting parallelism
- **Proposed method advantages**:
  - No intra-GPU contention
  - Maximum expert-level parallelism
  - Asynchronous communication hiding latency
  - Full GPU utilization for single expert computation

## 4. Discussion Points

### 4.1 Resource Utilization
- **Baseline**: 16 GPUs with shared resources and contention
- **Proposed**: 64 GPUs with dedicated resources per expert
- **Trade-off**: Higher GPU count for significantly improved performance

### 4.2 Communication Overhead
- **Network requirements**: High-bandwidth interconnects (NVLink, InfiniBand, NVSwitch)
- **Communication pattern**: Token routing between experts across nodes
- **Mitigation**: Asynchronous routing and compute-communication overlap

### 4.3 Scalability Characteristics
- **Linear scaling**: Achieved in large EP regime (EP ≥ 16)
- **Limiting factor**: Network bandwidth rather than compute
- **Future scaling**: Extensible to thousands of experts with sufficient network infrastructure

## 5. Validation Summary
- **Setting**: Inference-only evaluation
- **Model size**: Moderate (4-layer MoE with 64 total experts)
- **Hardware**: H100 cluster with high-speed interconnects
- **Results**: Significant performance improvement validates large EP approach
- **Reproducibility**: Clear configuration details provided for both baseline and proposed methods