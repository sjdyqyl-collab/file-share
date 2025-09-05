# Phase 3: Experiments Extraction

## Abstract (Retained from Original)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Experimental Setup

### Model Configuration
- **Model Type**: 4-layer Mixture-of-Experts (MoE)
- **Expert Architecture**: Multi-Layer Perceptron (MLP) per expert
- **Experts per Layer**: 16 experts
- **Total Experts**: 64 experts (4 layers × 16 experts/layer)
- **Precision**: FP16 (16-bit floating point)
- **Batch Size**: 1024 tokens per forward pass
- **Multi-Head Attention (MHA)**: 
  - Number of heads: 16
  - Dimension per head: 512
- **MLP Hidden Size**: 32,768 dimensions

### Hardware Configuration
- **GPU Type**: NVIDIA H100 GPUs
- **Baseline Deployment**: 16 H100 GPUs
- **Proposed Deployment**: 64 H100 GPUs
- **Environment**: High-performance computing (HPC) cluster
- **Network**: High-bandwidth, low-latency interconnects (NVLink, InfiniBand)

### Evaluation Metrics
- **TPS (Tokens per Second)**: Primary throughput metric
- **TPOT (Time per Output Token)**: Primary latency metric

## Parallel Deployment Details

### Baseline Configuration (TP=8, PP=2)
- **Parallel Strategy**: Tensor Parallelism (TP) + Pipeline Parallelism (PP)
- **GPU Allocation**: 16 H100 GPUs total
- **Tensor Parallelism**: TP=8 (each tensor split across 8 GPUs)
- **Pipeline Parallelism**: PP=2 (2 pipeline stages)
- **Per-GPU Deployment**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs
  - Experts are colocated: typically 4 experts per GPU
- **Processing Flow**: Tokens flow sequentially through pipeline stages
- **Resource Sharing**: Multiple experts per GPU share compute resources

### Proposed Cross-Node Expert Parallelism
- **Parallel Strategy**: Expert Parallelism (EP) + optional Tensor Parallelism (TP)
- **GPU Allocation**: 64 H100 GPUs total
- **Expert Parallelism**: EP=64 (one expert per GPU)
- **Per-GPU Deployment**:
  - Each GPU hosts exactly one expert
  - Tensor parallelism applied only if single expert exceeds GPU memory (optional TP=2)
  - Each MoE layer acts as a micro-stage
- **Token Routing**:
  - Input tokens dynamically routed to GPU holding corresponding expert
  - Token batches sent asynchronously to minimize idle time
  - Communication overlapped with computation
- **Resource Utilization**: All 64 experts per layer compute in parallel

## Experimental Results

### Performance Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### Performance Analysis
- **Throughput Improvement**: 3.75× increase (450,000 vs 120,000 TPS)
- **Latency Reduction**: 3.8× decrease (2.2ms vs 8.3ms TPOT)
- **Scaling Efficiency**: Near-linear scaling achieved with 64 GPUs
- **Resource Utilization**: Full GPU utilization through dedicated expert per GPU

### Bottleneck Analysis
- **Baseline Limitations**:
  - Intra-GPU contention between 4 experts per GPU
  - Pipeline stalls due to sequential processing
  - Shared compute resources limiting parallel execution
- **Proposed Advantages**:
  - No intra-GPU contention (1 expert per GPU)
  - Concurrent expert execution across all GPUs
  - Asynchronous communication hiding latency

## Experimental Validation

### Scalability Verification
- **EP=16 Regime**: Validated large expert parallelism (EP ≥ 16)
- **Network Efficiency**: Communication overhead effectively managed through:
  - Token batching optimization
  - Asynchronous routing implementation
  - Topology-aware placement

### Memory Usage
- **Per-Expert Memory**: Dedicated GPU memory allocation
- **No Sharing Overhead**: Eliminated memory contention between experts
- **Scalability**: Memory scales linearly with expert count

### Load Balancing Results
- **Dynamic Gating**: Successfully prevented expert overloading
- **Balanced Distribution**: Achieved uniform token distribution across experts
- **Straggler Prevention**: No significant performance degradation from slow experts

## Key Experimental Findings

1. **Large EP Effectiveness**: EP ≥ 16 successfully achieved with significant performance gains
2. **Communication Overhead Management**: Cross-node communication effectively overlapped with computation
3. **Resource Utilization**: One expert per GPU maximizes compute efficiency
4. **Scalability**: System scales near-linearly with additional GPUs
5. **Real-world Applicability**: Validated in HPC environment with modern GPU clusters