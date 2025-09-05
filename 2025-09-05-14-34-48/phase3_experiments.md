# Phase 3: Experiments and Results Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Model Type**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Total Experts**: 64 experts (4 layers × 16 experts/layer)
- **Expert Architecture**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (half precision)
- **Multi-Head Attention (MHA)**:
  - Number of heads: 16
  - Dimension per head: 512
- **MLP Hidden Size**: 32,768

### 1.2 Data Configuration
- **Batch Size**: 1024 sequences per batch
- **Sequence Length**: 10,000 tokens per sequence
- **Total Tokens per Batch**: 10,240,000 tokens (1024 × 10,000)

### 1.3 Hardware Environment
- **GPU Type**: H100 GPUs
- **Precision**: FP16
- **Setting**: Inference-only

### 1.4 Evaluation Metrics
- **TPS (Tokens per Second)**: Throughput measurement
- **TPOT (Time per Output Token)**: Latency per token measurement

## 2. Parallel Deployment Configurations

### 2.1 Baseline Configuration (TP=8, PP=2)
- **Total GPUs**: 16 H100 GPUs
- **Parallel Strategy**:
  - **Tensor Parallelism (TP)**: 8-way
  - **Pipeline Parallelism (PP)**: 2 stages
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (16 GPUs ÷ 2 stages)
  - **Experts per GPU**: 4 experts colocated on each GPU
  - **Expert Distribution**: 64 experts ÷ 16 GPUs = 4 experts/GPU
- **Processing Flow**:
  - Tokens flow sequentially through 2 pipeline stages
  - Multiple experts per GPU share compute resources
  - Intra-GPU contention between colocated experts

### 2.2 Proposed Cross-Node Expert Parallelism
- **Total GPUs**: 64 H100 GPUs
- **Parallel Strategy**:
  - **Expert Parallelism (EP)**: 64-way (EP=64)
  - **Tensor Parallelism (TP)**: Optional TP=2 if expert doesn't fit on single GPU
  - **Pipeline Parallelism (PP)**: Each MoE layer as micro-stage
- **Per-GPU Allocation**:
  - **Experts per GPU**: Exactly 1 expert per GPU
  - **Total Expert Coverage**: 64 experts across 64 GPUs
  - **Expert Placement**: One GPU per expert per layer
- **Processing Flow**:
  - **All 64 experts compute in parallel** across layers
  - No intra-GPU contention
  - Asynchronous token routing between experts
  - Communication overlapped with computation

## 3. Experimental Results

### 3.1 Performance Comparison

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### 3.2 Performance Analysis
- **Throughput Improvement**: 450,000 ÷ 120,000 = **3.75× higher TPS**
- **Latency Reduction**: 8.3 ÷ 2.2 = **3.8× lower TPOT**
- **GPU Utilization**: 
  - Baseline: 16 GPUs with shared resources
  - Proposed: 64 GPUs with dedicated resources
- **Resource Efficiency**: 
  - Baseline: 4× more experts per GPU → contention
  - Proposed: 1 expert per GPU → no contention

### 3.3 Scalability Characteristics
- **Linear Scaling**: Demonstrated for EP ≥ 16 regime
- **Communication Overhead**: Mitigated through asynchronous routing
- **Network Utilization**: High-bandwidth interconnects effectively utilized
- **Load Balancing**: Dynamic adjustment prevents expert overloading

## 4. Experimental Validation

### 4.1 Test Conditions
- **Environment**: H100 GPU cluster
- **Network**: NVLink/InfiniBand interconnects
- **Precision**: FP16 throughout
- **Batch Processing**: 1024 sequences consistently

### 4.2 Baseline Limitations Identified
- **Intra-GPU Contention**: 4 experts competing for same GPU resources
- **Pipeline Stalls**: Sequential processing through pipeline stages
- **Resource Underutilization**: GPU compute not fully utilized due to sharing
- **Memory Bandwidth**: Shared memory bandwidth across colocated experts

### 4.3 Proposed Method Advantages
- **Maximal Parallelism**: All 64 experts compute simultaneously
- **No Resource Contention**: Each expert has dedicated GPU
- **Overlapped Communication**: Tokens routed asynchronously
- **Near-Linear Scaling**: Performance scales with GPU count
- **Memory Efficiency**: Each expert has full GPU memory available

## 5. Detailed Resource Mapping

### 5.1 Expert-to-GPU Mapping (Proposed Method)
```
Layer 1: Expert 0-15   → GPUs 0-15   (Nodes 0-3, 4 GPUs per node)
Layer 2: Expert 16-31  → GPUs 16-31  (Nodes 4-7, 4 GPUs per node)
Layer 3: Expert 32-47  → GPUs 32-47  (Nodes 8-11, 4 GPUs per node)
Layer 4: Expert 48-63  → GPUs 48-63  (Nodes 12-15, 4 GPUs per node)
```

### 5.2 Communication Patterns
- **Inter-layer**: All-to-all communication for token redistribution
- **Intra-layer**: Point-to-point between token source and expert destination
- **Topology-aware**: Minimize cross-node traffic where possible

## 6. Reproducibility Details

### 6.1 Fixed Parameters
- **Model Architecture**: 4-layer MoE, 16 experts/layer
- **Sequence Length**: Fixed at 10,000 tokens
- **Precision**: FP16 consistently
- **Batch Size**: 1024 sequences

### 6.2 Variable Parameters
- **GPU Count**: 16 (baseline) vs 64 (proposed)
- **Parallel Strategy**: TP=8,PP=2 vs EP=64
- **Expert Density**: 4/GPU vs 1/GPU
- **Network Utilization**: Moderate vs intensive

### 6.3 Measurement Methodology
- **TPS Calculation**: Total tokens processed ÷ total time
- **TPOT Measurement**: Average time per output token across all tokens
- **Warmup**: Sufficient warmup iterations to stabilize measurements
- **Multiple Runs**: Results averaged over multiple experimental runs