# Experiments - Large-Scale Cross-Node Expert Parallelism

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 (total 64 experts)
- **Expert type**: Multi-Layer Perceptron (MLP)
- **Precision**: FP16 (half precision)
- **Batch size**: 1024 tokens per forward pass
- **Multi-Head Attention**: 
  - Number of heads: 16
  - Dimension per head: 512
  - Total attention dimension: 16 × 512 = 8192
- **MLP hidden size**: 32768 (feed-forward network dimension)

### 1.2 Hardware Configuration
- **GPU type**: NVIDIA H100
- **Baseline deployment**: 16 H100 GPUs
- **Proposed deployment**: 64 H100 GPUs
- **Network**: High-bandwidth interconnect (NVLink/InfiniBand)

### 1.3 Evaluation Metrics
- **TPS (Tokens per Second)**: Overall system throughput
- **TPOT (Time per Output Token)**: Latency per generated token

## 2. Baseline Deployment (TP=8, PP=2)

### 2.1 Parallelism Configuration
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2
- **Expert Parallelism (EP)**: Not explicitly used (experts colocated)
- **Total GPUs**: 16

### 2.2 GPU Allocation Details
- **Per-GPU allocation**:
  - 4 experts per GPU (16 experts × 4 layers ÷ 16 GPUs = 4 experts/GPU)
  - 1/8 tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (16 GPUs ÷ 2 stages = 8 GPUs/stage)
- **Expert placement**: Multiple experts colocated on same GPU
- **Memory usage**: Shared compute resources among experts on each GPU

### 2.3 Processing Flow
- **Sequential pipeline**: Tokens flow through 2 pipeline stages
- **Intra-GPU contention**: 4 experts share GPU compute units
- **Communication pattern**: 
  - TP=8: All-reduce within each TP group
  - PP=2: Send activations between pipeline stages

## 3. Proposed Cross-Node Expert Parallelism

### 3.1 Parallelism Configuration
- **Expert Parallelism (EP)**: 16 (minimum large EP)
- **Tensor Parallelism (TP)**: 1 (per expert, optional TP=2 if needed)
- **Pipeline Parallelism (PP)**: 4 (one stage per MoE layer)
- **Total GPUs**: 64 (one GPU per expert per layer)

### 3.2 GPU Allocation Details
- **Per-GPU allocation**:
  - Exactly 1 expert per GPU
  - 64 experts × 4 layers = 256 total expert instances
  - 256 expert instances ÷ 64 GPUs = 4 experts per GPU (one from each layer)
- **Expert placement**: Each GPU hosts one expert from each of the 4 layers
- **Memory isolation**: No shared compute resources between experts

### 3.3 Routing Implementation
- **Dynamic routing**: Tokens routed to GPU holding target expert
- **Asynchronous communication**: Token batches sent while computing
- **Overlap strategy**: 
  - Compute current batch while receiving next batch
  - NCCL all-to-all for token exchange

## 4. Performance Results

### 4.1 Throughput Comparison
| Method | GPUs Used | Deployment Strategy | TPS (Tokens/s) | TPOT (ms) | Improvement |
|--------|-----------|-------------------|----------------|-----------|-------------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 | - |
| Proposed Cross-Node | 64 | 1 expert per GPU | 450,000 | 2.2 | 3.75× TPS |

### 4.2 Detailed Analysis
- **Throughput gain**: 450,000 ÷ 120,000 = 3.75× improvement
- **Latency reduction**: 8.3 ÷ 2.2 = 3.77× improvement
- **GPU utilization**: 100% compute utilization per expert
- **Communication overhead**: Amortized by overlapping compute/comm

### 4.3 Scalability Characteristics
- **Linear scaling**: 4× GPUs (16→64) yields 3.75× throughput
- **Efficiency**: 93.75% scaling efficiency (3.75/4.0)
- **Bottleneck**: Network bandwidth in large EP regime

## 5. Resource Utilization

### 5.1 Memory Usage
- **Per-expert memory**:
  - MLP weights: 2 × 32768 × 8192 × 2 bytes = 1.07 GB (FP16)
  - Activations: 1024 × 8192 × 2 bytes = 16 MB per batch
- **Per-GPU memory**:
  - 4 experts × 1.07 GB = 4.28 GB for weights
  - 4 × 16 MB = 64 MB for activations
  - Total: ~4.35 GB per GPU (well within H100 memory)

### 5.2 Compute Utilization
- **Baseline**: GPU shared among 4 experts → ~25% per expert
- **Proposed**: GPU dedicated to 1 expert → 100% utilization
- **FLOPS**: 32768 × 8192 × 2 = 536.9 GFLOPs per expert forward pass

## 6. Network Communication

### 6.1 Communication Patterns
- **Baseline**:
  - TP=8: All-reduce within 8-GPU groups
  - PP=2: Send 1024 × 8192 × 2 = 16 MB between stages
- **Proposed**:
  - EP=16: All-to-all token exchange
  - Token volume: 1024 tokens × 8192 dim × 2 bytes = 16 MB per expert
  - Total: 16 × 16 MB = 256 MB all-to-all communication

### 6.2 Overlap Efficiency
- **Compute time**: 536.9 GFLOPs ÷ 989 TFLOPs (H100) ≈ 0.54 ms
- **Communication time**: 16 MB ÷ 50 GB/s (NVLink) ≈ 0.32 ms
- **Overlap ratio**: 0.32/0.54 ≈ 59% communication hidden

## 7. Experimental Validations

### 7.1 Load Balance Verification
- **Expert utilization**: Uniform distribution across 64 experts
- **Token routing**: Balanced load via dynamic gating
- **Straggler mitigation**: No single expert bottleneck observed

### 7.2 Network Saturation Test
- **Bandwidth utilization**: ~60% of available NVLink bandwidth
- **Latency impact**: <5% increase due to cross-node communication
- **Scalability limit**: Linear up to 64 GPUs tested