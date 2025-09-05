# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Total experts**: 64 experts (4 layers × 16 experts/layer)
- **Expert type**: Each expert is a Multi-Layer Perceptron (MLP)
- **Precision**: FP16 (half precision floating point)
- **Batch size**: 1024 tokens per forward pass
- **Multi-Head Attention (MHA)**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total attention dimension: 16 × 512 = 8192
- **MLP hidden dimension**: 32768

### 1.2 Hardware Setup
- **GPU type**: NVIDIA H100 GPUs
- **Tested configurations**:
  - Baseline: 16 H100 GPUs
  - Proposed method: 64 H100 GPUs

### 1.3 Evaluation Metrics
- **TPS (Tokens per Second)**: Measures overall throughput
- **TPOT (Time per Output Token)**: Measures latency per generated token

## 2. Parallel Deployment Configurations

### 2.1 Baseline Deployment (TP=8, PP=2)
- **Total GPUs**: 16 H100
- **Parallelism strategy**:
  - **Tensor Parallelism (TP)**: 8-way
  - **Pipeline Parallelism (PP)**: 2 stages
- **Per-GPU allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (2 stages × 8 GPUs = 16 total)
  - **Experts per GPU**: 4 experts (64 total experts ÷ 16 GPUs)
  - **Expert placement**: Colocated on GPUs
- **Processing flow**: Tokens flow sequentially through pipeline stages
- **Resource sharing**: Multiple experts per GPU share compute resources

### 2.2 Proposed Cross-Node Expert Parallelism
- **Total GPUs**: 64 H100
- **Parallelism strategy**:
  - **Expert Parallelism (EP)**: 64-way (maximum possible)
  - **Tensor Parallelism (TP)**: Optional TP=2 if expert FFN exceeds GPU memory
  - **Pipeline Parallelism (PP)**: Each MoE layer as micro-stage
- **Per-GPU allocation**:
  - **Experts per GPU**: Exactly 1 expert per GPU
  - **Total coverage**: All 64 experts (4 layers × 16 experts) distributed
  - **Memory usage**: Tensor parallelism applied only when necessary
- **Routing mechanism**:
  - **Dynamic routing**: Input tokens routed to GPU holding corresponding expert
  - **Asynchronous transfer**: Token batches sent asynchronously
  - **Overlap**: Communication overlapped with computation

## 3. Experimental Results

### 3.1 Performance Comparison Table

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### 3.2 Performance Analysis
- **Throughput improvement**: 450,000 ÷ 120,000 = **3.75× higher TPS**
- **Latency reduction**: 8.3 ÷ 2.2 = **3.8× lower TPOT**
- **GPU utilization**: Full utilization of all 64 GPUs vs shared resources in baseline
- **Scaling efficiency**: Near-linear scaling demonstrated with 64 GPUs

### 3.3 Bottleneck Analysis
- **Baseline bottlenecks**:
  - Intra-GPU expert contention (4 experts sharing GPU)
  - Pipeline stalls due to sequential processing
  - Limited expert-level parallelism
- **Proposed method advantages**:
  - No expert contention (1 expert per GPU)
  - Concurrent expert processing
  - Minimal idle time through async communication

## 4. Experimental Validation

### 4.1 Test Environment
- **Setting**: Inference-only evaluation
- **Network**: High-bandwidth interconnects (NVLink/InfiniBand)
- **Precision**: FP16 throughout
- **Batch consistency**: 1024 tokens maintained across configurations

### 4.2 Scalability Demonstration
- **Large EP regime**: EP=64 (≥16 qualifies as large EP)
- **Resource scaling**: Linear increase from 16 to 64 GPUs
- **Performance scaling**: 3.75× throughput with 4× GPUs (near-linear)

## 5. Experimental Parameters Summary

### 5.1 Fixed Parameters
- **Model layers**: 4
- **Experts per layer**: 16
- **Total experts**: 64
- **Precision**: FP16
- **Batch size**: 1024 tokens
- **MLP hidden**: 32768
- **Attention**: 16 heads × 512 dim = 8192

### 5.2 Variable Parameters
- **GPU count**: 16 (baseline) vs 64 (proposed)
- **Experts per GPU**: 4 (baseline) vs 1 (proposed)
- **Parallelism degrees**:
  - Baseline: TP=8, PP=2
  - Proposed: EP=64, optional TP=2

### 5.3 Performance Metrics
- **Baseline**: 120k TPS, 8.3ms TPOT
- **Proposed**: 450k TPS, 2.2ms TPOT
- **Improvement ratios**: 3.75× throughput, 3.8× latency reduction