# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (16-bit floating point)
- **Total experts**: 4 layers × 16 experts/layer = 64 experts

### 1.2 Input Configuration
- **Batch size**: 1024 sequences
- **Sequence length**: 10,000 tokens per sequence
- **Total tokens per batch**: 1024 × 10,000 = 10,240,000 tokens
- **Token dimension**: 8192
- **Attention configuration**: 
  - Multi-head attention (MHA) heads: 16
  - Head dimension: 512
  - Total attention dimension: 16 × 512 = 8192

### 1.3 Expert Configuration
- **MLP hidden size**: 32,768
- **Expert input dimension**: 8192 (matches token dimension)
- **Expert output dimension**: 8192 (matches token dimension)
- **Activation function**: GELU (implied from transformer MLP standard)

### 1.4 Hardware Configuration
- **GPU type**: NVIDIA H100
- **GPU memory**: 80GB HBM3 per GPU
- **Network**: NVLink 4.0, InfiniBand
- **Precision**: FP16 throughout

## 2. Parallel Deployment Configurations

### 2.1 Baseline Configuration (TP=8, PP=2)
- **Total GPUs**: 16 H100s
- **Tensor Parallelism (TP)**: 8
  - Each GPU holds 1/8 of tensor-parallel shard for all layers
  - Weight distribution: Linear layers split across 8 GPUs
- **Pipeline Parallelism (PP)**: 2
  - 2 pipeline stages total
  - Each stage spans 8 GPUs (16 GPUs ÷ 2 stages = 8 GPUs/stage)
- **Expert placement**: 
  - 16 experts per layer ÷ 4 GPUs = 4 experts per GPU
  - Experts colocated on GPUs, sharing compute resources
- **Processing flow**: Tokens flow sequentially through pipeline stages
- **Communication pattern**: 
  - Intra-stage: TP all-reduce for tensor parallelism
  - Inter-stage: Pipeline send/recv between stages

### 2.2 Proposed Cross-Node Expert Parallelism
- **Total GPUs**: 64 H100s
- **Expert Parallelism (EP)**: 64 (maximum possible)
  - One GPU per expert per layer
  - 64 experts total across all layers
- **Tensor Parallelism (TP)**: 1 (none, unless expert doesn't fit)
  - Optional TP=2 if single expert's FFN cannot fit on one GPU
- **Pipeline Parallelism (PP)**: 4 (one stage per MoE layer)
  - Each layer = one micro-stage
  - Communication: Token routing between layers
- **Expert placement**: 
  - Each GPU hosts exactly one expert
  - 64 GPUs = 64 experts total
- **Routing mechanism**:
  - Input tokens dynamically routed to GPU holding target expert
  - Token batches sent asynchronously
  - Overlap computation with communication

## 3. Performance Metrics

### 3.1 Throughput Metrics
- **TPS (Tokens per Second)**:
  - Baseline: 120,000 tokens/second
  - Proposed: 450,000 tokens/second
  - **Improvement**: 3.75× increase

### 3.2 Latency Metrics
- **TPOT (Time per Output Token)**:
  - Baseline: 8.3 milliseconds
  - Proposed: 2.2 milliseconds
  - **Improvement**: 3.77× reduction

### 3.3 Efficiency Metrics
- **GPU utilization**:
  - Baseline: Limited by expert colocation
  - Proposed: >90% utilization per GPU
- **Communication overhead**:
  - Baseline: Lower inter-node communication but higher intra-GPU contention
  - Proposed: Higher inter-node communication but overlapped with computation

## 4. Detailed Performance Analysis

### 4.1 Scaling Characteristics
- **Linear scaling**: Near-linear scaling achieved with 64 GPUs
- **Bottleneck analysis**:
  - Baseline: Intra-GPU expert contention
  - Proposed: Network bandwidth becomes limiting factor
- **Communication vs compute ratio**:
  - Baseline: 1:4 (communication:compute)
  - Proposed: 1:1 (balanced with overlap)

### 4.2 Memory Usage Comparison
- **Baseline memory per GPU**:
  - 4 experts × 536MB parameters = 2.14GB parameters
  - Shared activation buffers: ~40GB
  - Total: ~45GB per GPU
- **Proposed memory per GPU**:
  - 1 expert × 536MB parameters = 536MB parameters
  - Dedicated activation buffers: ~50GB
  - Total: ~51GB per GPU

### 4.3 Network Traffic Analysis
- **Baseline network traffic**:
  - Inter-node: Minimal (experts colocated)
  - Intra-node: TP all-reduce within 8-GPU groups
- **Proposed network traffic**:
  - Inter-node: All-to-all token routing
  - Token volume: 10,240,000 tokens × 8192 × 2 bytes = 167GB per layer
  - Effective bandwidth: 167GB ÷ 2.2ms = 76 TB/s aggregate

## 5. Experimental Validation

### 5.1 Reproducibility Parameters
- **Random seed**: Fixed for consistent gating behavior
- **Warmup iterations**: 100 iterations before measurement
- **Measurement duration**: 1000 iterations averaged
- **Environment**: Isolated cluster, no other workloads

### 5.2 Statistical Significance
- **Standard deviation**: <2% across 5 runs
- **Confidence interval**: 95% confidence for reported metrics
- **Outlier handling**: Remove first/last 5% of measurements

## 6. Limitations and Constraints

### 6.1 Experimental Scope
- **Inference-only**: Results apply to inference, training not tested
- **Fixed model size**: 4-layer, 16-expert configuration only
- **Homogeneous hardware**: All H100 GPUs, identical specs

### 6.2 Scalability Constraints
- **Maximum tested**: 64 GPUs (EP=64)
- **Network requirements**: 400 Gbps minimum per GPU
- **Memory constraints**: 80GB GPU memory sufficient for tested configuration