# Experiments - Large-Scale Cross-Node Expert Parallelism

## 1. Experimental Setup

### Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Expert type**: Each expert is a Multi-Layer Perceptron (MLP)
- **Precision**: FP16 (half precision)

### Input Configuration
- **Batch size**: 1024 sequences per batch
- **Sequence length**: 10,000 tokens per sequence
- **Token dimension**: 8,192 dimensions per token

### Attention Configuration
- **Multi-Head Attention (MHA)**: 
  - Number of heads: 16
  - Dimension per head: 512
  - Total MHA dimension: 16 × 512 = 8,192

### Expert Configuration
- **MLP hidden size**: 32,768
- **Expert type**: Feed-forward network (FFN) replacement

### Evaluation Metrics
- **TPS (Tokens per Second)**: Measures overall throughput
- **TPOT (Time per Output Token)**: Measures latency per token

## 2. Parallel Deployment Details

### 2.1 Baseline Deployment (TP=8, PP=2)
- **GPUs Used**: 16 H100 GPUs
- **Parallelism Configuration**:
  - Tensor Parallelism (TP): 8-way
  - Pipeline Parallelism (PP): 2 stages
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (2 stages total)
  - Experts are colocated: typically 4 experts per GPU
- **Processing Flow**:
  - Tokens flow sequentially through pipeline stages
  - Multiple experts per GPU share compute resources
  - Intra-GPU contention occurs due to expert colocation

### 2.2 Proposed Cross-Node Expert Parallelism
- **GPUs Used**: 64 H100 GPUs
- **Per-GPU Allocation**:
  - Each GPU hosts exactly **one expert**
  - Total experts: 16 experts/layer × 4 layers = 64 experts
  - Tensor Parallelism: Optional TP=2 only if single expert's FFN cannot fit on one GPU
  - Pipeline Parallelism: Each MoE layer acts as a micro-stage
- **Expert Distribution**:
  - One expert per GPU across all 64 GPUs
  - All 64 experts per layer compute in parallel
  - No expert colocation, eliminating intra-GPU contention
- **Routing Mechanism**:
  - Input tokens dynamically routed to GPU holding corresponding expert
  - Token batches sent asynchronously to minimize idle time
  - Cross-node communication overlapped with computation

## 3. Experimental Results

### Performance Comparison Table
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### Performance Improvements
- **Throughput**: 3.75× improvement (450,000 vs 120,000 TPS)
- **Latency**: 3.8× reduction (2.2ms vs 8.3ms TPOT)
- **GPU Utilization**: 4× more GPUs used (64 vs 16)
- **Efficiency**: Near-linear scaling achieved in large EP regime

### Key Observations
1. **Baseline Limitations**:
   - GPUs shared among multiple experts causing intra-GPU contention
   - Pipeline stalls due to sequential processing
   - Limited expert-level parallelism

2. **Proposed Method Advantages**:
   - Dedicated GPU per expert eliminates resource contention
   - Maximal expert-level parallelism achieved
   - Asynchronous routing minimizes waiting time
   - Near-linear scaling with 64 GPUs in large EP regime (EP ≥ 16)

## 4. Discussion
- **Resource Trade-off**: 4× more GPUs yield 3.75× throughput improvement
- **Scalability**: System scales near-linearly with unlimited H100 GPUs
- **Communication Overhead**: Effectively mitigated through asynchronous routing
- **Deployment Feasibility**: Practical in HPC environments with abundant GPU resources

## 5. Experimental Configuration Summary
- **Hardware**: H100 GPUs (16 vs 64)
- **Model**: 4-layer MoE, 16 experts/layer, FP16
- **Large EP regime**: EP = 16 (minimum threshold)
- **Key metric**: 3.75× throughput improvement validates large-scale expert parallelism approach