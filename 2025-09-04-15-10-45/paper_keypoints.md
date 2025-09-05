# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models - Key Points

## Abstract (Retained Original)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Technical Contributions

### 1. Core Innovation
- **Single-expert-per-GPU deployment**: Each GPU hosts at most one expert
- **Large EP regime**: EP ≥ 16 experts per parallel group
- **Cross-node distribution**: Experts distributed across nodes for maximum parallelism

### 2. Expert Placement Strategy
- **Topology-aware placement**: Considers node-to-node bandwidth, latency, GPU memory
- **One-expert-per-GPU principle**: Minimizes resource contention
- **Dynamic replication**: When E > G, replicate experts to balance memory

### 3. Routing and Load Balancing
- **Token batching**: Group tokens by destination expert
- **Asynchronous routing**: Overlap communication with computation
- **Dynamic load balancing**: Adjust gating probabilities to prevent overloading

### 4. Communication Optimization
- **Compute-communication overlap**: Use CUDA streams/NCCL for async operations
- **Pipeline scheduling**: Immediate routing between MoE layers
- **Topology-aware routing**: Minimize cross-node traffic

## Experimental Results

### Setup
- **Model**: 4-layer MoE, 16 experts per layer, MLP experts
- **Precision**: FP16
- **Batch size**: 1024 tokens
- **MHA**: 16 heads × 512 dim per head
- **MLP hidden size**: 32768

### Performance Comparison
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert per GPU | 450,000 | 2.2ms |

### Key Metrics
- **3.75× higher throughput** (450k vs 120k TPS)
- **3.8× lower latency** (2.2ms vs 8.3ms TPOT)
- **Near-linear scaling** with 64 GPUs

## Model Configuration Summary
- **Layers**: 4
- **Experts per layer**: 16 (total 64 experts)
- **Expert type**: MLP with hidden size 32768
- **Attention**: 16 heads, 512 dim per head
- **Precision**: FP16
- **Batch size**: 1024 tokens per forward pass