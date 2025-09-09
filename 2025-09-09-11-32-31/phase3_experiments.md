# Phase 3: Experiments Extraction

## Experimental Setup

### Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (16-bit floating point)
- **Batch Configuration**: 
  - 1024 sequences per batch
  - 10,000 tokens per sequence
  - Total: 10.24M tokens per batch
- **Token Dimension**: 8192
- **Multi-Head Attention (MHA)**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total MHA dimension: 8192 (16 × 512)
- **MLP Hidden Size**: 32,768

### Hardware Environment
- **GPU Type**: H100 (NVIDIA Hopper architecture)
- **Setting**: Inference-only (no training)
- **Interconnect**: High-performance networking (NVLink, InfiniBand, NVSwitch fabrics)

### Evaluation Metrics
- **TPS (Tokens per Second)**: Primary throughput measurement
- **TPOT (Time per Output Token)**: Latency measurement per token

## Parallel Deployment Details

### Baseline Deployment (TP=8, PP=2)
- **Total GPUs**: 16 H100
- **Parallelism Configuration**:
  - Tensor Parallelism (TP): 8-way
  - Pipeline Parallelism (PP): 2 stages
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (2 stages total)
  - Expert colocation: 4 experts per GPU
- **Processing Flow**: Tokens flow sequentially through pipeline stages with shared compute resources

### Proposed Cross-Node Expert Parallelism
- **Total GPUs**: 64 H100
- **Parallelism Configuration**:
  - Expert Parallelism (EP): 64 (one expert per GPU)
  - Optional Tensor Parallelism (TP): 2-way (only if single expert FFN cannot fit)
  - Pipeline Parallelism: Each MoE layer as micro-stage
- **Per-GPU Allocation**:
  - Each GPU hosts exactly one expert
  - No expert colocation (strict one-expert-per-GPU policy)
- **Routing Mechanism**:
  - Dynamic token routing to GPU holding corresponding expert
  - Asynchronous token batch transfer
  - Overlapped communication with computation

## Results

### Performance Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### Performance Improvements
- **Throughput Gain**: 3.75× improvement (450,000 vs 120,000 TPS)
- **Latency Reduction**: 3.8× reduction (2.2ms vs 8.3ms TPOT)
- **GPU Utilization**: 4× more GPUs (64 vs 16) yielding 3.75× throughput (near-linear scaling)

### Detailed Analysis

#### Baseline Characteristics
- **Resource Sharing**: Multiple experts per GPU cause intra-GPU contention
- **Pipeline Stalls**: Sequential processing through pipeline stages
- **Compute Bottleneck**: GPU compute units shared among 4 experts per GPU
- **Memory Pressure**: Higher memory usage per GPU due to expert colocation

#### Proposed Method Characteristics
- **Resource Isolation**: Each expert has dedicated GPU resources
- **Parallel Processing**: All 64 experts per layer compute simultaneously
- **Communication Overlap**: Asynchronous routing minimizes idle time
- **Memory Efficiency**: Lower per-GPU memory usage with single expert
- **Network Optimization**: Topology-aware placement and token batching

### Scalability Analysis
- **Linear Scaling**: 4× GPU increase yielding 3.75× throughput (93.75% efficiency)
- **Large EP Regime**: EP=64 qualifies as large EP (≥16) with demonstrated benefits
- **Network Impact**: Communication overhead successfully mitigated through overlap techniques
- **Future Scalability**: Blueprint for deployments with thousands of experts

## Discussion

### Key Success Factors
1. **Expert Isolation**: One expert per GPU eliminates computational contention
2. **Asynchronous Communication**: Overlaps token transfer with computation
3. **Load Balancing**: Prevents expert stragglers and network hotspots
4. **Topology Awareness**: Optimizes expert placement for network characteristics

### Limitations and Considerations
- **GPU Requirement**: Requires 4× more GPUs than baseline
- **Network Dependency**: Relies on high-bandwidth, low-latency interconnects
- **Inference-Only**: Results demonstrated for inference; training implications need exploration
- **Model Size**: Current evaluation on 4-layer model; scaling to deeper networks needs validation

### Practical Implications
- **HPC Environments**: Particularly effective in clusters with abundant GPU resources
- **Cost-Benefit**: 3.75× performance gain may justify 4× GPU cost in latency-critical applications
- **Future-Proofing**: Aligns with trend toward larger GPU clusters and advanced networking

## Experimental Reproducibility

### Key Parameters for Replication
- **Model Dimensions**: 4 layers, 16 experts/layer, 8192 token dim, 32768 MLP hidden
- **Batch Settings**: 1024 sequences × 10000 tokens
- **Hardware**: H100 GPUs with high-performance interconnects
- **Parallelism**: EP=64 vs TP=8, PP=2 baseline
- **Metrics**: TPS and TPOT measurements under inference-only conditions