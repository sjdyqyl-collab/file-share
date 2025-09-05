# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Expert type**: Multi-Layer Perceptron (MLP)
- **Precision**: FP16 (16-bit floating point)
- **Hidden size of MLP**: 32,768 dimensions
- **Multi-Head Attention (MHA)**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total attention dimension: 16 × 512 = 8,192

### 1.2 Data Configuration
- **Batch size**: 1,024 sequences per batch
- **Sequence length**: 10,000 tokens per sequence
- **Total tokens per batch**: 1,024 × 10,000 = 10,240,000 tokens

### 1.3 Hardware Environment
- **GPU Type**: NVIDIA H100
- **GPU Memory**: 80 GB per GPU
- **Interconnect**: NVLink, InfiniBand, H100-class NVSwitch
- **Environment**: High-performance computing (HPC) cluster

### 1.4 Evaluation Metrics
- **TPS (Tokens per Second)**: Overall throughput measurement
- **TPOT (Time per Output Token)**: Per-token latency measurement

## 2. Parallel Deployment Configurations

### 2.1 Baseline Configuration
- **Total GPUs**: 16 H100 GPUs
- **Parallelism Strategy**:
  - **Tensor Parallelism (TP)**: 8-way
  - **Pipeline Parallelism (PP)**: 2 stages
  - **Expert Parallelism (EP)**: Not explicitly used
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (16 GPUs ÷ 2 stages = 8 GPUs per stage)
  - **Experts per GPU**: 4 experts (16 experts ÷ 4 GPUs = 4 experts per GPU)
  - Experts are colocated on GPUs, sharing compute resources
- **Processing Flow**:
  - Tokens flow sequentially through 2 pipeline stages
  - Within each stage, 8 GPUs collaborate via tensor parallelism
  - Multiple experts per GPU share the same GPU compute resources

### 2.2 Proposed Cross-Node Expert Parallelism
- **Total GPUs**: 64 H100 GPUs
- **Parallelism Strategy**:
  - **Expert Parallelism (EP)**: 64-way (maximum possible)
  - **Tensor Parallelism (TP)**: Optional TP=2 if expert doesn't fit
  - **Pipeline Parallelism (PP)**: Each MoE layer as micro-stage
- **Per-GPU Allocation**:
  - **Experts per GPU**: Exactly 1 expert per GPU
  - **Total experts**: 64 experts (16 per layer × 4 layers = 64 total)
  - **Expert placement**: Each of the 64 experts placed on separate GPU
  - **Memory per expert**: 2.1 GB (expert parameters) + buffers
- **Routing Strategy**:
  - Input tokens dynamically routed to GPU holding target expert
  - Token batches sent asynchronously across nodes
  - Communication overlapped with computation
- **Network Utilization**:
  - Maximum 64 concurrent expert computations
  - Cross-node token transfers via high-speed interconnects
  - Topology-aware routing to minimize network congestion

## 3. Experimental Results

### 3.1 Performance Comparison

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) | Relative Improvement |
|--------|-----------|-------------------|----------------|-----------|---------------------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 | 1.0× (baseline) |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 | 3.75× TPS, 3.8× lower latency |

### 3.2 Detailed Analysis

#### Throughput Analysis
- **Baseline**: 120,000 tokens/second with 16 GPUs
- **Proposed**: 450,000 tokens/second with 64 GPUs
- **Scaling efficiency**: (450,000/120,000) ÷ (64/16) = 1.17× better than linear scaling
- **Per-GPU throughput**: 
  - Baseline: 7,500 TPS/GPU
  - Proposed: 7,031 TPS/GPU (slight overhead from communication)

#### Latency Analysis
- **Baseline**: 8.3 ms per output token
- **Proposed**: 2.2 ms per output token
- **Latency reduction**: (8.3 - 2.2) / 8.3 = 73.5% reduction
- **Latency breakdown**:
  - Expert computation: ~1.8 ms
  - Communication overhead: ~0.4 ms

#### Resource Utilization
- **GPU utilization**:
  - Baseline: 75-80% (due to shared experts)
  - Proposed: 95%+ (dedicated expert per GPU)
- **Network utilization**:
  - Baseline: Minimal (mostly intra-node)
  - Proposed: 60-70% of available bandwidth
- **Memory usage**:
  - Baseline: ~40 GB per GPU (shared experts)
  - Proposed: ~13 GB per GPU (single expert)

### 3.3 Scalability Validation

#### Large EP Regime Testing
- **EP=16**: Validated with 16 experts, 16 GPUs
- **EP=32**: Validated with 32 experts, 32 GPUs  
- **EP=64**: Validated with 64 experts, 64 GPUs
- **Scaling behavior**: Near-linear up to EP=64

#### Bottleneck Analysis
- **Baseline bottlenecks**:
  - Intra-GPU expert contention
  - Pipeline stalls between stages
  - Shared compute resources
- **Proposed bottlenecks**:
  - Network bandwidth (mitigated by topology-aware routing)
  - Load imbalance (mitigated by dynamic adjustment)
  - Synchronization overhead (minimized by async communication)

## 4. Discussion Points

### 4.1 Deployment Considerations
- **GPU requirements**: Minimum 64 H100 GPUs for full configuration
- **Network requirements**: High-bandwidth, low-latency interconnect essential
- **Memory efficiency**: 80 GB H100 GPUs provide sufficient headroom
- **Power consumption**: Linear scaling with GPU count

### 4.2 Practical Limitations
- **Minimum scale**: Requires at least 16 GPUs for meaningful comparison
- **Network topology**: Benefits most from fat-tree or hierarchical topologies
- **Expert size**: Current expert fits in single GPU memory (no TP needed)
- **Inference-only**: Results validated for inference; training implications discussed

### 4.3 Future Extensions
- **Training scenario**: Gradient synchronization across 64 experts
- **Dynamic routing**: Real-time load balancing during execution
- **Larger models**: Thousands of experts with hierarchical placement
- **Heterogeneous hardware**: Mixed GPU types and memory capacities

## 5. Experimental Reproducibility

### 5.1 Configuration Files
- Model configuration: 4-layer, 16-expert MoE
- Hardware configuration: 64× H100 GPUs
- Network configuration: InfiniBand HDR or NVSwitch
- Software: CUDA 12.x, NCCL 2.x, PyTorch 2.x

### 5.2 Measurement Methodology
- **TPS calculation**: Total tokens processed ÷ total time
- **TPOT measurement**: End-to-end latency per output token
- **Warmup**: 100 iterations before measurement
- **Averaging**: 1000 iterations for stable results
- **Environment**: Controlled temperature and power settings