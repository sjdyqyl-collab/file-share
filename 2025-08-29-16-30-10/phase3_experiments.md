# Phase 3: Experiments Extraction

## Experimental Setup

### Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (16-bit floating point)
- **Batch Size**: 1024 tokens per forward pass
- **Multi-Head Attention**: 16 heads, 512 dimensions per head
- **MLP Hidden Size**: 32768

### Hardware Environment
- **GPU Type**: H100 GPUs
- **Setting**: Inference-only
- **Scale**: 16 GPUs (baseline) vs 64 GPUs (proposed)

### Evaluation Metrics
- **TPS (Tokens per Second)**: Measures overall throughput
- **TPOT (Time per Output Token)**: Measures latency per token

## Parallel Deployment Details

### 2.1 Baseline Deployment (TP=8, PP=2)
- **Total GPUs**: 16 H100 GPUs
- **Tensor Parallelism (TP)**: 8-way
- **Pipeline Parallelism (PP)**: 2 stages
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (2 stages × 8 GPUs = 16 total)
  - Expert colocation: 4 experts per GPU (16 experts ÷ 4 GPUs per stage)
- **Processing Flow**: Tokens flow sequentially through pipeline stages, with multiple experts sharing compute resources on each GPU

### 2.2 Proposed Cross-Node Expert Parallelism
- **Total GPUs**: 64 H100 GPUs
- **Expert Parallelism (EP)**: 64 (one GPU per expert per layer)
- **Per-GPU Allocation**:
  - Each GPU hosts exactly **one expert**
  - Tensor parallelism applied only if single expert's FFN exceeds GPU memory (optional TP=2)
  - Pipeline parallelism: Each MoE layer acts as micro-stage with overlapped communication
- **Routing Strategy**:
  - Input tokens dynamically routed to GPU holding corresponding expert
  - Token batches sent asynchronously to minimize idle time
- **Parallelism Achievement**: All 64 experts per layer compute in parallel

## Experimental Results

### Performance Comparison Table
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### Performance Analysis
- **Throughput Improvement**: 3.75× higher (450,000 vs 120,000 TPS)
- **Latency Reduction**: 3.8× lower (2.2ms vs 8.3ms TPOT)
- **Scalability**: Near-linear scaling with 64 GPUs in large EP regime (EP ≥ 16)

### Key Performance Factors
1. **Expert Isolation**: One expert per GPU eliminates intra-GPU contention
2. **Parallel Execution**: All 64 experts compute simultaneously
3. **Communication Overlap**: Asynchronous token routing minimizes waiting
4. **Resource Utilization**: Full GPU compute utilization through expert isolation

## Discussion Points

### Baseline Limitations
- **Intra-GPU Contention**: 4 experts sharing single GPU creates computational bottlenecks
- **Pipeline Stalls**: Sequential processing through pipeline stages introduces delays
- **Resource Sharing**: Multiple experts competing for same GPU resources

### Proposed Method Advantages
- **Compute Isolation**: Each expert has dedicated GPU resources
- **Maximal Parallelism**: All experts process tokens simultaneously
- **Communication Efficiency**: Asynchronous routing with computation overlap
- **Scalability**: Demonstrates near-linear scaling with abundant GPU resources

### Large EP Regime Validation
- **EP=64 Configuration**: Successfully demonstrates large expert parallelism
- **Network Efficiency**: Modern HPC networking sustains high bandwidth/low latency
- **Compute Saturation**: Full GPU utilization achieved through expert distribution