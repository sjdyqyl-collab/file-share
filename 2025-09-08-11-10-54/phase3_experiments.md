# Phase 3: Experiments Extraction

## Experimental Setup

### 1. Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Total Experts**: 64 experts (4 layers × 16 experts)
- **Expert Type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (16-bit floating point)

### 2. Input Configuration
- **Batch Size**: 1024 sequences per batch
- **Sequence Length**: 10000 tokens per sequence
- **Token Dimension**: 8192 dimensions per token
- **Total Tokens per Batch**: 10,240,000 tokens (1024 × 10000)

### 3. Model Dimensions
- **MHA Configuration**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total MHA dimension: 8192 (16 × 512)
- **MLP Hidden Size**: 32768 dimensions
- **Expert FFN Structure**: Standard transformer feed-forward network

### 4. Hardware Configuration
- **GPU Type**: NVIDIA H100
- **Experiment 1 (Baseline)**: 16 H100 GPUs
- **Experiment 2 (Proposed)**: 64 H100 GPUs
- **Setting**: Inference-only evaluation

## Parallel Deployment Details

### 5. Baseline Deployment (TP=8, PP=2)
- **Total GPUs**: 16 H100
- **Parallel Strategy**:
  - Tensor Parallelism (TP): 8
  - Pipeline Parallelism (PP): 2
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs
  - 2 total pipeline stages
  - **Experts per GPU**: 4 experts colocated on each GPU
- **Processing Flow**:
  - Tokens flow sequentially through pipeline stages
  - Multiple experts per GPU share compute resources
  - Expert-level contention due to colocation

### 6. Proposed Cross-Node Expert Parallelism
- **Total GPUs**: 64 H100
- **Parallel Strategy**:
  - Expert Parallelism (EP): 64 (one expert per GPU)
  - Tensor Parallelism (TP): Optional TP=2 if single expert cannot fit
  - Pipeline Parallelism: Each MoE layer as micro-stage
- **Per-GPU Allocation**:
  - Each GPU hosts exactly one expert
  - No expert colocation
  - Maximum expert-level parallelism achieved
- **Routing Strategy**:
  - Input tokens dynamically routed to GPU holding corresponding expert
  - Token batches sent asynchronously
  - Overlapping communication with computation

## Results

### 7. Performance Metrics

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) | Improvement |
|--------|-----------|-------------------|----------------|-----------|-------------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 | - |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 | 3.75× TPS, 3.8× lower latency |

### 8. Detailed Analysis

#### 8.1 Throughput Analysis
- **Baseline Throughput**: 120,000 tokens/second
- **Proposed Throughput**: 450,000 tokens/second
- **Improvement Factor**: 3.75× higher throughput
- **Throughput per GPU**: 
  - Baseline: 7,500 tokens/s/GPU (120,000/16)
  - Proposed: 7,031 tokens/s/GPU (450,000/64)

#### 8.2 Latency Analysis
- **Baseline Latency**: 8.3 ms per token (TPOT)
- **Proposed Latency**: 2.2 ms per token (TPOT)
- **Improvement Factor**: 3.8× lower latency
- **Latency Reduction**: 6.1 ms absolute reduction

#### 8.3 Resource Utilization
- **Baseline Issues**:
  - Intra-GPU contention from 4 experts sharing resources
  - Pipeline stalls due to sequential processing
  - Limited expert-level parallelism
- **Proposed Benefits**:
  - Full GPU compute utilization per expert
  - No expert contention on same device
  - Maximum concurrent expert processing

### 9. Scalability Validation

#### 9.1 Large EP Regime Performance
- **EP=64** configuration successfully demonstrates large-scale expert parallelism
- **Near-linear scaling** achieved with 64 GPUs
- **Communication overhead** effectively mitigated through:
  - Asynchronous token routing
  - Topology-aware placement
  - Overlapped computation and communication

#### 9.2 Network Efficiency
- **Cross-node communication** sustained by H100-class interconnects
- **Bandwidth utilization** optimized through token batching
- **Latency hiding** achieved via CUDA streams and asynchronous operations

## Discussion

### 10. Key Insights
- **Expert Isolation**: Deploying one expert per GPU eliminates resource contention
- **Parallelism Maximization**: EP=64 enables unprecedented expert-level concurrency
- **Communication Trade-offs**: Network bandwidth becomes primary consideration, not compute
- **Modern Hardware Leverage**: H100 NVLink/InfiniBand capabilities fully utilized

### 11. Limitations and Considerations
- **GPU Requirement**: Requires 4× more GPUs than baseline (64 vs 16)
- **Network Dependency**: Performance critically dependent on high-bandwidth interconnects
- **Inference Focus**: Results validated for inference-only scenarios
- **Model Size**: Specific to 64-expert configuration, may need adjustment for different scales

### 12. Reproducibility Specifications
- **Hardware**: NVIDIA H100 GPUs with high-bandwidth interconnects
- **Software**: CUDA streams, NCCL/MPI for communication
- **Precision**: FP16 throughout for consistent comparison
- **Batch Configuration**: Fixed 1024 sequences × 10000 tokens for fair evaluation