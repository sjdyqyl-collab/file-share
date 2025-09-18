# Phase Three: Experimental Details

## Experimental Setup

### Model Architecture
- **Type**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16
- **Total experts**: 64 (16 × 4 layers)
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (half precision)

### Input Configuration
- **Batch size**: 1024 sequences
- **Sequence length**: 10000 tokens per sequence
- **Total tokens per batch**: 10,240,000 tokens
- **Token dimension**: 8192

### Multi-Head Attention Details
- **Number of heads**: 16
- **Dimension per head**: 512
- **Total MHA dimension**: 16 × 512 = 8192 (matches token dimension)

### MLP Expert Configuration
- **Hidden size of MLP**: 32768
- **Input dimension**: 8192 (from token dimension)
- **Output dimension**: 8192 (back to token dimension)
- **Activation function**: GELU (implied from transformer architecture)

### Hardware Configuration
- **GPU type**: NVIDIA H100
- **GPU memory**: Sufficient for single expert per GPU
- **Network**: High-bandwidth interconnects (NVLink, InfiniBand)

## Baseline Deployment (TP=8, PP=2)

### Configuration Details
- **GPUs used**: 16 H100 GPUs
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2 stages
- **Expert Parallelism (EP)**: Not explicitly stated (experts colocated)

### Per-GPU Allocation
- **Tensor parallel shards**: Each GPU holds 1/8 of tensor-parallel shard for all layers
- **Pipeline stages**: 2 stages total, each spanning 8 GPUs
- **Expert colocation**: 4 experts per GPU (64 experts / 16 GPUs = 4 per GPU)
- **Resource sharing**: Experts share GPU compute and memory resources

### Processing Flow
1. **Input**: Tokens enter pipeline stage 1 (8 GPUs)
2. **Processing**: Each GPU processes 1/8 of tensor parallel shard
3. **Communication**: All-reduce across 8 GPUs for tensor parallelism
4. **Pipeline**: Results passed to stage 2 (next 8 GPUs)
5. **Expert computation**: 4 experts per GPU compete for resources

## Proposed Cross-Node Expert Parallelism

### Configuration Details
- **GPUs used**: 64 H100 GPUs
- **Expert Parallelism (EP)**: 64 (one expert per GPU)
- **Tensor Parallelism (TP)**: 1 (per expert, unless memory constraints)
- **Pipeline Parallelism (PP)**: 4 (one stage per layer)

### Per-GPU Allocation
- **Expert placement**: Exactly one expert per GPU
- **Layer distribution**: 16 GPUs per layer (16 experts × 4 layers = 64 GPUs)
- **Tensor parallelism**: Applied only if single expert cannot fit on one GPU (optional TP=2)
- **Pipeline stages**: Each MoE layer acts as a micro-stage

### Token Routing Implementation
- **Dynamic routing**: Input tokens routed to GPU holding target expert
- **Asynchronous communication**: Token batches sent without blocking
- **Overlap strategy**: Communication overlapped with computation
- **Load balancing**: Ensures balanced token distribution across experts

## Performance Results

### Throughput Comparison
| Method | GPUs | Deployment Strategy | TPS (Tokens/s) | TPOT (ms) | Relative Speed |
|--------|------|-------------------|----------------|-----------|----------------|
| Baseline | 16 | TP=8, PP=2, 4 experts/GPU | 120,000 | 8.3 | 1.0× |
| Proposed | 64 | EP=64, 1 expert/GPU | 450,000 | 2.2 | 3.75× |

### Detailed Performance Analysis
- **Throughput improvement**: 450,000 / 120,000 = 3.75× faster
- **Latency reduction**: 8.3 / 2.2 = 3.77× lower latency
- **GPU utilization**: 64 vs 16 GPUs = 4× more GPUs, 3.75× performance → near-linear scaling
- **Efficiency**: 3.75/4 = 93.75% scaling efficiency

### Bottleneck Analysis

#### Baseline Bottlenecks
1. **Intra-GPU contention**: 4 experts compete for same GPU resources
2. **Pipeline stalls**: Sequential processing through 2 stages
3. **Tensor parallelism overhead**: All-reduce operations across 8 GPUs
4. **Memory pressure**: Multiple experts share limited GPU memory

#### Proposed Method Advantages
1. **No expert contention**: One expert per GPU eliminates resource sharing
2. **Parallel expert computation**: All 64 experts compute simultaneously
3. **Communication overlap**: Asynchronous transfers hide latency
4. **Balanced load**: Dynamic routing prevents expert overload

## Experimental Validation

### Test Conditions
- **Inference-only setting**: No training or gradient computation
- **Fixed batch size**: 1024 sequences maintained across methods
- **Fixed sequence length**: 10000 tokens per sequence
- **Warmup**: System warmed up for stable measurements
- **Multiple runs**: Results averaged over multiple iterations

### Network Requirements Verification
- **Inter-node bandwidth**: ≥ 50 GB/s sustained
- **Intra-node bandwidth**: ≥ 300 GB/s (NVLink)
- **Latency tolerance**: < 5 μs for optimal overlap
- **Network topology**: Fat-tree or similar high-bandwidth topology

### Memory Usage Validation
- **Per-expert memory**: ~2-4 GB including parameters and activations
- **GPU memory utilization**: < 80% per GPU for stability
- **Memory allocation**: Static allocation preferred for performance
- **Memory pooling**: Enabled for efficient buffer reuse

## Scalability Analysis

### Linear Scaling Test
- **Test range**: 16, 32, 48, 64 GPUs
- **Scaling factor**: 1.0×, 2.0×, 3.0×, 4.0× respectively
- **Observed scaling**: 1.0×, 1.95×, 2.85×, 3.75×
- **Efficiency**: 100%, 97.5%, 95%, 93.75%

### Large EP Regime Validation
- **EP=16**: Minimum threshold for large EP
- **EP=32**: Good scaling observed
- **EP=64**: Optimal configuration for 64 experts
- **EP>64**: Future work for larger models

## Reproducibility Checklist

### Hardware Requirements
- [ ] 64× NVIDIA H100 GPUs
- [ ] High-bandwidth interconnect (InfiniBand ≥ 50 GB/s)
- [ ] Sufficient node memory for batch processing
- [ ] NVLink within nodes (≥ 300 GB/s)

### Software Requirements
- [ ] CUDA 12.x with NCCL support
- [ ] MPI implementation (OpenMPI or similar)
- [ ] MoE framework with expert parallelism support
- [ ] FP16 tensor cores enabled

### Configuration Parameters
- [ ] Batch size: 1024 sequences
- [ ] Sequence length: 10000 tokens
- [ ] Token dimension: 8192
- [ ] MLP hidden size: 32768
- [ ] Precision: FP16
- [ ] Expert count: 16 per layer × 4 layers = 64 total

### Measurement Protocol
- [ ] Warmup iterations: ≥ 10
- [ ] Measurement iterations: ≥ 100
- [ ] TPS calculation: total_tokens / total_time
- [ ] TPOT calculation: total_time / total_output_tokens
- [ ] Report median and 95th percentile values