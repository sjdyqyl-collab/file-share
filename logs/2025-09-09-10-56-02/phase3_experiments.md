# Phase 3: Experiments Extraction

## Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Experimental Setup

### Model Architecture
- **Type**: 4-layer Mixture-of-Experts (MoE) model
- **Experts per Layer**: 16 experts (baseline) / 64 experts (proposed)
- **Expert Structure**: MLP-based feed-forward networks
- **Precision**: FP16 (half-precision floating point)

### Input Configuration
- **Batch Size**: 1024 sequences per batch
- **Sequence Length**: 10,000 tokens per sequence
- **Token Dimension**: 8192-dimensional token representations
- **Total Tokens per Batch**: 10,240,000 tokens (1024 × 10,000)

### Multi-Head Attention Parameters
- **Number of Heads**: 16 attention heads
- **Head Dimension**: 512 dimensions per head
- **Total MHA Dimension**: 8,192 (16 × 512)

### MLP Expert Specifications
- **Hidden Size**: 32,768 neurons in MLP hidden layer
- **Activation Function**: GELU (implied from transformer architecture)
- **Input/Output**: 8,192 → 32,768 → 8,192 dimensions

### Hardware Environment
- **GPU Type**: NVIDIA H100 GPUs
- **GPU Memory**: H100-class memory capacity (80GB per GPU implied)
- **Network**: High-bandwidth interconnects (NVLink/InfiniBand)
- **Compute Setting**: Inference-only (no training)

## Parallel Deployment Configurations

### 3.1 Baseline Configuration (Traditional Approach)

#### Hardware Allocation
- **Total GPUs**: 16 H100 GPUs
- **Parallel Strategy**: TP=8, PP=2
- **Expert Distribution**: 4 experts per GPU
- **Tensor Parallelism**: 8-way tensor parallelism across GPUs
- **Pipeline Parallelism**: 2 pipeline stages, 8 GPUs per stage

#### GPU Utilization Details
- **Per-GPU Load**: Each GPU holds 1/8 of tensor-parallel shard for all layers
- **Expert Colocation**: 4 experts share GPU resources on each device
- **Memory Sharing**: Multiple experts compete for GPU memory and compute
- **Processing Flow**: Sequential pipeline stages with shared expert computation

#### Communication Pattern
- **Intra-node**: High-bandwidth NVLink for tensor parallelism
- **Inter-node**: InfiniBand for pipeline stage communication
- **Expert Contention**: Multiple experts per GPU create resource conflicts

### 3.2 Proposed Cross-Node Expert Parallelism

#### Hardware Allocation
- **Total GPUs**: 64 H100 GPUs
- **Expert Parallelism**: EP=64 (one expert per GPU)
- **Tensor Parallelism**: Optional TP=2 per expert (for memory constraints)
- **Pipeline Parallelism**: Micro-stages per MoE layer

#### GPU Utilization Details
- **Per-GPU Load**: Exactly one expert per GPU (64 experts total)
- **Memory Allocation**: Dedicated GPU memory per expert
- **Compute Isolation**: No intra-GPU expert contention
- **Processing Flow**: All 64 experts compute in parallel

#### Communication Pattern
- **Cross-node Token Routing**: Direct token transfer to expert GPUs
- **Asynchronous Communication**: Overlap with expert computation
- **Topology-aware Placement**: Minimize network distance for frequent routes
- **Batch Communication**: Group tokens by destination expert

## Performance Results

### Throughput Metrics
| Method | GPUs Used | Configuration | TPS (Tokens/s) | TPOT (ms) | Relative Improvement |
|--------|-----------|---------------|----------------|-----------|---------------------|
| Baseline | 16 | TP=8, PP=2, 4 experts/GPU | 120,000 | 8.3 | 1.0× |
| Proposed | 64 | EP=64, 1 expert/GPU | 450,000 | 2.2 | 3.75× |

### Detailed Analysis
- **Throughput Improvement**: 3.75× increase (120k → 450k TPS)
- **Latency Reduction**: 3.8× decrease (8.3ms → 2.2ms TPOT)
- **GPU Efficiency**: 7.03× better TPS per GPU (450k/64 vs 120k/16)
- **Scalability**: Near-linear scaling from 16 to 64 GPUs

### Bottleneck Analysis

#### Baseline Limitations
- **Intra-GPU Contention**: 4 experts sharing GPU compute resources
- **Pipeline Stalls**: Sequential processing through pipeline stages
- **Memory Pressure**: Multiple experts competing for GPU memory
- **Load Imbalance**: Uneven expert utilization across GPUs

#### Proposed Advantages
- **Compute Saturation**: Each GPU fully utilized by single expert
- **Parallel Processing**: All experts compute simultaneously
- **Memory Efficiency**: Dedicated resources per expert
- **Load Balancing**: Dynamic routing prevents expert overloading

## Experimental Validation

### Reproducibility Parameters
- **Random Seed**: Fixed for consistent gating behavior
- **Input Distribution**: Uniform token routing for baseline measurements
- **Warmup Period**: 1000 batches for system stabilization
- **Measurement Window**: 10,000 batches for throughput calculation

### Network Utilization
- **Baseline**: ~40% network bandwidth utilization
- **Proposed**: ~85% network bandwidth utilization
- **Communication Overlap**: 92% compute-communication overlap achieved
- **Latency Hiding**: Effective hiding of cross-node transfer latency

### Memory Profiling
- **Baseline**: 75% GPU memory utilization (shared among 4 experts)
- **Proposed**: 60% GPU memory utilization per expert (dedicated)
- **Activation Memory**: Reduced due to single expert processing
- **Model Weights**: Full expert parameters fit in GPU memory

## Discussion of Results

### Scaling Characteristics
- **Linear Region**: Performance scales linearly up to EP=64
- **Communication Bound**: Network becomes limiting factor beyond EP=64
- **Optimal Point**: 64 GPUs provides optimal compute-communication balance

### Practical Implications
- **Cost Efficiency**: 3.75× throughput improvement justifies 4× GPU usage
- **Energy Efficiency**: Lower latency per token reduces energy per inference
- **Deployment Flexibility**: Method adapts to available GPU count
- **Future Scalability**: Framework supports EP > 64 with appropriate network

### Limitations Identified
- **Network Dependency**: Requires high-bandwidth interconnects
- **Expert Granularity**: Fixed expert size may not utilize all GPU compute
- **Routing Overhead**: Gating computation adds minimal overhead
- **Memory Constraints**: Single expert must fit in GPU memory