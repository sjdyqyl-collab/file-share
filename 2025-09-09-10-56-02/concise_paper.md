# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## 1. Introduction
Traditional MoE parallelization assigns multiple experts to the same GPU, creating computational bottlenecks that limit expert-level parallelism. We present a cross-node expert parallelism method that distributes experts such that each GPU hosts at most one expert, pushing EP ≥ 16 to maximize concurrent computation. This shifts optimization from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities.

## 2. Methodology

### 2.1 Expert Placement Strategy
- **Single-expert-per-GPU**: At most one expert per GPU eliminates intra-GPU contention
- **Cross-node distribution**: Topology-aware placement considering bandwidth, latency, and memory
- **Large EP regime**: EP ≥ 16 for maximum expert independence

### 2.2 Routing and Load Balancing
- **Token batching**: Group tokens by destination expert to minimize network messages
- **Asynchronous routing**: Send token batches asynchronously while overlapping computation
- **Dynamic load balancing**: Monitor per-expert load and adjust gating probabilities

### 2.3 Communication Overlap
- **Dual stream architecture**: Separate compute and communication streams
- **Pipeline scheduling**: Micro-stages per MoE layer with immediate token routing
- **CUDA streams**: 4 streams for concurrent operations with NCCL/MPI integration

### 2.4 Memory Integration
- **Tensor parallelism**: Optional TP=2 per expert if memory constrained
- **Data parallelism**: Synchronized weight updates across replicas
- **Dedicated resources**: Each expert has isolated GPU memory

## 3. Experiments

### 3.1 Setup
- **Model**: 4-layer MoE, 16/64 experts per layer, MLP-based experts
- **Precision**: FP16
- **Input**: 1024 sequences × 10,000 tokens × 8192 dimensions
- **Hardware**: H100 GPUs with high-bandwidth interconnects

### 3.2 Configurations
| Method | GPUs | Configuration | TPS | TPOT |
|--------|------|---------------|-----|------|
| Baseline | 16 | TP=8, PP=2, 4 experts/GPU | 120,000 | 8.3ms |
| Proposed | 64 | EP=64, 1 expert/GPU | 450,000 | 2.2ms |

### 3.3 Results
- **3.75× throughput improvement** (120k → 450k TPS)
- **3.8× latency reduction** (8.3ms → 2.2ms TPOT)
- **7.03× efficiency per GPU** improvement
- **Near-linear scaling** from 16 to 64 GPUs

## 4. Deployment Configuration

### 4.1 Baseline (16 GPUs)
- **Parallelism**: TP=8, PP=2, EP=1
- **Expert placement**: 4 experts per GPU across 2 pipeline stages
- **Memory**: 25GB model weights + 45GB activations per GPU
- **Communication**: NVLink intra-node, InfiniBand inter-node

### 4.2 Proposed (64 GPUs)
- **Parallelism**: EP=64, TP=1, PP=1 (micro-stages)
- **Expert placement**: 1 expert per GPU, topology-aware distribution
- **Memory**: 12GB expert weights + 8GB activations + 6GB buffers per GPU
- **Communication**: 400Gbps InfiniBand with 92% compute-communication overlap

## 5. Conclusion
Large-scale cross-node expert parallelism with EP ≥ 16 achieves 3.75× higher throughput and 3.8× lower latency by dedicating one expert per GPU and overlapping communication with computation. This provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.

## Key Technical Parameters
- **Model dimensions**: 8192 token dim, 32768 MLP hidden, 16×512 MHA
- **Batch processing**: 1024 sequences × 10k tokens
- **Precision**: FP16 throughout
- **Network**: 400Gbps+ InfiniBand required
- **Optimal scale**: 64 H100 GPUs for 4-layer, 64-expert MoE