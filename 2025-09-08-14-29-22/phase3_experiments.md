# Phase 3: Experiments Extraction

## Experimental Setup

### Model Configuration
- **Model Type**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Architecture**: Each expert is a Multi-Layer Perceptron (MLP)
- **Precision**: FP16
- **Batch Configuration**: 
  - 1024 sequences per batch
  - 10000 tokens per sequence
- **Token Dimension**: 8192
- **Multi-Head Attention (MHA)**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total MHA dimension: 16 × 512 = 8192
- **MLP Hidden Size**: 32768

### Evaluation Metrics
- **TPS (Tokens per Second)**: Measures overall throughput
- **TPOT (Time per Output Token)**: Measures latency per token

## Parallel Deployment Details

### Baseline Deployment (TP=8, PP=2)
- **GPUs Used**: 16 H100 GPUs
- **Parallel Strategy**:
  - Tensor Parallelism (TP): 8-way
  - Pipeline Parallelism (PP): 2 stages
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (2 stages total = 16 GPUs)
  - Experts are colocated on GPUs: typically 4 experts per GPU
- **Processing Flow**: Tokens flow sequentially through pipeline stages, multiple experts per GPU share compute resources

### Proposed Cross-Node Expert Parallelism
- **GPUs Used**: 64 H100 GPUs
- **Parallel Strategy**:
  - Expert Parallelism (EP): 64-way (16 experts/layer × 4 layers = 64 total experts)
  - Tensor Parallelism (TP): Optional TP=2 if single expert's FFN cannot fit on one GPU
  - Pipeline Parallelism: Each MoE layer is a micro-stage
- **Per-GPU Allocation**:
  - Each GPU hosts exactly one expert
  - Communication of tokens is overlapped with computation
- **Routing Strategy**:
  - Input tokens dynamically routed to GPU holding corresponding expert
  - Token batches sent asynchronously ensuring minimal idle time
- **Expert Distribution**: All 64 experts per layer compute in parallel (maximizing throughput)

## Results

### Performance Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### Performance Improvements
- **Throughput**: 3.75× higher (450,000 vs 120,000 TPS)
- **Latency**: 3.8× lower (2.2ms vs 8.3ms TPOT)

### Key Advantages of Proposed Method
- Dedicated one expert per GPU eliminates intra-GPU contention
- Full utilization of all 64 GPUs for expert-level parallelism
- Asynchronous token routing minimizes waiting across nodes
- Near-linear scaling in large EP regime (EP ≥ 16)

## Discussion Points
- One expert per GPU allows full utilization of GPU compute and memory
- Asynchronous token routing ensures minimal waiting even across nodes
- With 64 GPUs (unlimited H100s), system scales near-linearly in large EP regime
- Trade-off: increased communication overhead vs. maximized compute concurrency