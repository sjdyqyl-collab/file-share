# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (16-bit floating point)
- **Total experts**: 64 experts (4 layers × 16 experts/layer)

### 1.2 Input Specifications
- **Batch size**: 1024 sequences per batch
- **Sequence length**: 10,000 tokens per sequence
- **Token dimension**: 8,192 dimensions per token
- **Total tokens per batch**: 10,240,000 tokens (1024 × 10,000)

### 1.3 Attention Configuration
- **Multi-Head Attention (MHA)**: 16 attention heads
- **Head dimension**: 512 dimensions per head
- **Total attention dimension**: 8,192 (16 × 512)

### 1.4 MLP Configuration
- **Hidden size**: 32,768 dimensions
- **Activation function**: Not specified (typically GELU or ReLU in MoE)

### 1.5 Hardware Environment
- **GPU type**: NVIDIA H100
- **Inference-only setting**: No training performed
- **Network**: High-bandwidth interconnects (NVLink/InfiniBand)

## 2. Parallel Deployment Configurations

### 2.1 Baseline Configuration (TP=8, PP=2)
- **Total GPUs**: 16 H100 GPUs
- **Tensor Parallelism (TP)**: 8-way
- **Pipeline Parallelism (PP)**: 2 stages
- **Expert Parallelism (EP)**: Not explicitly used (experts colocated)

#### 2.1.1 GPU Allocation Details
- **Per-GPU allocation**:
  - 1/8 of tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (16 GPUs ÷ 2 stages)
  - 4 experts per GPU (64 total experts ÷ 16 GPUs)
- **Processing flow**: Tokens flow sequentially through pipeline stages
- **Resource contention**: Multiple experts per GPU share compute resources

### 2.2 Proposed Cross-Node Expert Parallelism
- **Total GPUs**: 64 H100 GPUs
- **Expert Parallelism (EP)**: 64-way (EP ≥ 16, qualifying as "large EP")
- **Tensor Parallelism (TP)**: Optional TP=2 if single expert exceeds GPU memory
- **Pipeline Parallelism (PP)**: Each MoE layer as micro-stage

#### 2.2.1 GPU Allocation Details
- **Per-GPU allocation**: Exactly one expert per GPU
- **Expert distribution**: 64 experts across 64 GPUs (1:1 mapping)
- **Memory usage**: Each GPU hosts complete expert parameters
- **Communication**: Tokens routed asynchronously to destination GPUs

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
- **Throughput improvement**: 3.75× (450,000 ÷ 120,000)
- **Latency reduction**: 3.77× (8.3 ÷ 2.2)
- **GPU utilization**: 4× more GPUs (64 vs 16) yielding 3.75× throughput
- **Efficiency**: 93.75% scaling efficiency (3.75 ÷ 4.0)

## 4. Experimental Validation

### 4.1 Scalability Verification
- **Linear scaling**: Near-linear scaling achieved with EP ≥ 16
- **Communication overhead**: Effectively mitigated through asynchronous routing
- **Load balancing**: Dynamic adjustment prevents expert overloading

### 4.2 Resource Utilization
- **GPU compute**: Fully utilized with one expert per GPU
- **Memory bandwidth**: Optimized for single expert processing
- **Network bandwidth**: High-bandwidth interconnects sustain cross-node communication

### 4.3 Bottleneck Analysis
- **Baseline bottlenecks**:
  - Intra-GPU contention from 4 experts sharing resources
  - Pipeline stalls between stages
  - Limited expert-level parallelism
- **Proposed method advantages**:
  - No intra-GPU expert contention
  - Maximal expert-level parallelism
  - Overlapped communication and computation

## 5. Reproducibility Details

### 5.1 Model Parameters
- **Total parameters**: Not explicitly stated, but calculable:
  - Attention layers: 4 × (8192 × 8192 × 3 + 8192 × 8192) ≈ 1.6B parameters
  - Expert MLPs: 64 × (8192 × 32768 + 32768 × 8192) ≈ 34.4B parameters
  - **Total**: ~36B parameters

### 5.2 Memory Requirements
- **Per expert**: 8192 × 32768 × 2 (weights) × 2 (FP16) ≈ 1GB per expert
- **Per GPU**: ~1GB for expert + attention shard + activations
- **Total memory**: 64GB across all experts

### 5.3 Communication Patterns
- **Token routing**: 8,192 dimensional tokens sent between nodes
- **Batch communication**: 1024 tokens per expert per batch
- **Bandwidth requirement**: ~32GB/s per link for full overlap