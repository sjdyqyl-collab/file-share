# Phase 3: Experiments Extraction

## Experimental Setup

### Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Total experts**: 64 experts (16 × 4 layers)
- **Expert type**: Each expert is a Multi-Layer Perceptron (MLP)
- **Precision**: FP16 (half precision)
- **Batch size**: 1024 sequences per batch
- **Sequence length**: 10,000 tokens per sequence
- **Token dimension**: 8,192 dimensions per token
- **Multi-Head Attention (MHA)**: 16 heads, 512 dimensions per head
- **MLP hidden size**: 32,768 dimensions

### Evaluation Metrics
- **TPS (Tokens per Second)**: Measures overall throughput of the system
- **TPOT (Time per Output Token)**: Measures latency per token in milliseconds

## Parallel Deployment Details

### Baseline Configuration (TP=8, PP=2)
- **GPUs used**: 16 H100 GPUs
- **Per-GPU allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage (2 stages total) spans 8 GPUs
  - Experts are colocated on GPUs, typically 4 experts per GPU
- **Processing flow**: Tokens flow sequentially through the pipeline stages, and multiple experts per GPU share compute resources

### Proposed Cross-Node Expert Parallelism
- **GPUs used**: 64 H100 GPUs (one GPU per expert per layer)
- **Per-GPU allocation**:
  - Each GPU hosts **exactly one expert**
  - Tensor parallelism is applied only if a single expert's FFN cannot fit on one GPU (optional TP=2)
  - Pipeline parallelism: each MoE layer is a micro-stage; communication of tokens is overlapped with computation
- **Routing mechanism**:
  - Input tokens are dynamically routed to the GPU holding the corresponding expert
  - Token batches are asynchronously sent, ensuring minimal idle time
- **Expert distribution**: All 64 experts per layer compute in parallel, maximizing throughput and minimizing token latency

## Results

### Performance Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### Performance Improvements
- **Throughput improvement**: 3.75× higher (450,000 vs 120,000 tokens/second)
- **Latency reduction**: 3.8× lower latency (2.2ms vs 8.3ms per token)
- **GPU utilization**: 4× more GPUs used (64 vs 16) with better per-GPU efficiency

### Key Observations
- **Baseline limitations**: GPUs are shared among multiple experts, causing intra-GPU contention and pipeline stalls
- **Proposed method advantages**: Dedicates one expert per GPU, enabling maximal expert-level parallelism
- **Scalability**: With 64 GPUs, the system scales near-linearly in the large EP regime (EP ≥ 16)

## Discussion

### Performance Analysis
- **Intra-GPU contention elimination**: By dedicating one expert per GPU, the proposed method eliminates resource sharing conflicts
- **Communication-computation overlap**: Asynchronous token routing ensures minimal waiting, even across nodes
- **Network utilization**: Modern HPC networking capabilities (NVLink, InfiniBand, NVSwitch) sustain high bandwidth and low latency

### Scalability Implications
- **Linear scaling**: The approach demonstrates near-linear scaling when moving from 16 to 64 GPUs
- **Resource efficiency**: Better utilization of available GPU resources compared to traditional colocation strategies
- **Future applicability**: Provides a scalable blueprint for high-performance MoE inference in large GPU clusters

## Experimental Limitations
- **Inference-only setting**: Results are demonstrated for inference, not training
- **Hardware specificity**: Results obtained on H100 GPUs with high-performance interconnects
- **Model size**: 4-layer MoE with 64 total experts - may not generalize to much larger models without additional considerations