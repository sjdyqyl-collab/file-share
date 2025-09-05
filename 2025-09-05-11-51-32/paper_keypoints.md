# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models - Key Points

## Abstract (Retained Original)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Innovation Points

### 1. Core Concept
- **One expert per GPU**: Deploy at most one expert per GPU to maximize expert-level parallelism
- **Large EP regime**: EP ≥ 16 for maximum independence of expert computation
- **Cross-node distribution**: Distribute experts across nodes to exploit distributed resources

### 2. Problem Solved
- **Expert contention**: Eliminates intra-GPU contention from multiple experts sharing resources
- **Scalability bottleneck**: Overcomes limitations of traditional MoE parallelization strategies
- **Compute efficiency**: Shifts bottleneck from computation to communication (which can be optimized)

### 3. Technical Components

#### Expert Placement Strategy
- Single-expert-per-GPU deployment
- Topology-aware placement considering bandwidth, latency, memory capacity
- Replication strategy when E > G (number of experts > GPUs)

#### Routing and Load Balancing
- Token batching by destination expert
- Asynchronous routing to overlap with computation
- Dynamic load balancing to prevent expert overloading

#### Communication Overlap
- Interleaved computation and communication
- CUDA streams/NCCL for asynchronous transfers
- Fine-grained pipeline scheduling across layers

### 4. Integration with Other Parallelisms
- Tensor Model Parallelism (TP): Applied within expert if needed
- Data Parallelism (DP): Applied across MoE network replicas
- Pipeline Parallelism (PP): Each MoE layer as micro-stage

## Experimental Validation

### Model Configuration
- 4-layer MoE, 16 experts per layer
- FP16 precision
- Batch: 1024 sequences × 10000 tokens
- MHA: 16 heads × 512 dim per head
- MLP hidden: 32768

### Results Comparison
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert per GPU | 450,000 | 2.2ms |

### Performance Gains
- **3.75× higher throughput** (450k vs 120k tokens/s)
- **3.8× lower latency** (2.2ms vs 8.3ms per token)
- Near-linear scaling in large EP regime

## Critical Dimensions and Parameters
- EP ≥ 16 (large EP threshold)
- One expert per GPU constraint
- 64 experts per layer in experiment
- 32768 MLP hidden dimension
- 1024 batch size, 10000 sequence length
- FP16 precision throughout

## Deployment Requirements
- HPC networking (NVLink, InfiniBand, NVSwitch)
- Topology-aware placement algorithms
- Asynchronous communication libraries (NCCL/MPI)
- Dynamic load balancing mechanisms