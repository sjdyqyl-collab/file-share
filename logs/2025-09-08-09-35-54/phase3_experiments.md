# Phase 3: Experiments Extraction

## Experimental Setup

### Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16
- **Batch Size**: 1024 sequences
- **Sequence Length**: 10000 tokens per sequence
- **Token Dimension**: 8192
- **MHA Configuration**: 16 heads, 512 dimensions per head
- **Hidden Size of MLP**: 32768

### Metrics
- **TPS (Tokens per Second)**: Measures throughput
- **TPOT (Time per Output Token)**: Measures latency per token

## Parallel Deployment Details

### 2.1 Baseline Deployment (TP=8, PP=2)
- **GPUs Used**: 16 H100
- **Configuration**:
  - Tensor Parallelism (TP): 8-way
  - Pipeline Parallelism (PP): 2 stages
  - Expert Parallelism (EP): 4 experts per GPU
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs
  - Experts are colocated on GPUs: 4 experts per GPU
- **Processing**: Tokens flow sequentially through pipeline stages, multiple experts per GPU share compute resources

### 2.2 Proposed Cross-Node Expert Parallelism
- **GPUs Used**: 64 H100
- **Configuration**:
  - Expert Parallelism (EP): 64 (one GPU per expert per layer)
  - Tensor Parallelism (TP): Optional TP=2 if single expert exceeds GPU memory
  - Pipeline Parallelism: Each MoE layer as a micro-stage
- **Per-GPU Allocation**:
  - Each GPU hosts **exactly one expert**
  - Token communication overlapped with computation
- **Routing**:
  - Input tokens dynamically routed to GPU holding corresponding expert
  - Token batches asynchronously sent to minimize idle time

## Results

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

## Performance Analysis
- **Throughput Improvement**: 3.75× higher (450,000 vs 120,000 TPS)
- **Latency Reduction**: 3.8× lower (2.2ms vs 8.3ms TPOT)
- **GPU Utilization**: Full utilization of all 64 GPUs vs shared resources in baseline
- **Scalability**: Near-linear scaling in large EP regime (EP ≥ 16)

## Key Findings
- **Single-expert-per-GPU** eliminates intra-GPU contention
- **Asynchronous token routing** minimizes waiting across nodes
- **Cross-node expert distribution** enables maximal expert-level parallelism
- **Communication-compute overlap** sustains high throughput despite increased network traffic