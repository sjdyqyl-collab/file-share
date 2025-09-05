# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## 1. Introduction
Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, scaling MoE models across large GPU clusters introduces challenges in expert placement and parallelization. Traditional approaches colocate multiple experts per GPU to reduce communication, creating computational bottlenecks that limit expert-level parallelism.

We present a cross-node expert parallelism method that distributes experts such that each GPU hosts at most one expert, pushing EP to 16 or beyond. This maximizes concurrent computation, leveraging modern HPC networking capabilities to sustain high bandwidth and low latency.

## 2. Methods

### 2.1 Expert Placement Strategy
- **Single-expert-per-GPU**: Each GPU hosts at most one expert
- **Cross-node distribution**: Topology-aware placement considering bandwidth, latency, and memory
- **Mathematical constraint**: For E experts and G GPUs, ensure E ≤ G for unique assignment

### 2.2 Routing and Load Balancing
- **Token batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous routing**: Overlap token transfer with expert computation
- **Dynamic load balancing**: Adjust gating probabilities based on per-expert load

### 2.3 Communication Overlap
- **Compute-communication overlap**: Use CUDA streams for asynchronous operations
- **Pipeline scheduling**: Each MoE layer as a micro-stage with immediate token routing
- **Fine-grained synchronization**: Start processing partial batches upon arrival

### 2.4 Large EP Regime (EP ≥ 16)
- **Definition**: Expert Parallelism degree of 16 or more
- **Network optimization**: Token batching reduces messages by 16×
- **Scalability**: Near-linear scaling with proper load balancing

## 3. Experiments

### 3.1 Setup
- **Model**: 4-layer MoE, 16 experts per layer, MLP experts
- **Precision**: FP16
- **Batch**: 1024 tokens per forward pass
- **Hardware**: H100 GPUs (80GB HBM3)
- **Metrics**: TPS (Tokens/Second), TPOT (ms/token)

### 3.2 Configurations

| Method | GPUs | Parallelism | Expert Placement | TPS | TPOT |
|--------|------|-------------|------------------|-----|------|
| Baseline | 16 | TP=8, PP=2 | 4 experts/GPU | 120,000 | 8.3ms |
| Proposed | 64 | EP=64 | 1 expert/GPU | 450,000 | 2.2ms |

### 3.3 Results
- **Throughput improvement**: 3.75× higher (450k vs 120k TPS)
- **Latency reduction**: 3.8× lower (2.2ms vs 8.3ms TPOT)
- **Scaling efficiency**: 93.75% (450k/480k theoretical max)
- **Resource utilization**: 95%+ GPU compute, 75% network bandwidth

## 4. Deployment Configuration

### 4.1 Model Dimensions
- **Experts per layer**: 16
- **Total layers**: 4
- **Expert hidden size**: 32768
- **MHA**: 16 heads × 512 dimensions = 8192 total
- **Memory per GPU**: ~50GB (single expert)

### 4.2 Hardware Requirements
- **Baseline**: 2 nodes × 8 GPUs (16 total)
- **Proposed**: 8 nodes × 8 GPUs (64 total)
- **Network**: InfiniBand NDR 400 Gbps, NVLink 900 GB/s

### 4.3 Software Stack
- **Framework**: PyTorch 2.0 with custom MoE kernels
- **Communication**: NCCL 2.18+, MPI
- **Precision**: FP16 compute, FP32 master weights
- **CUDA**: 12.0+

## 5. Conclusion
Our large-scale cross-node expert parallelism method achieves 3.75× higher throughput and 3.8× lower latency by dedicating one expert per GPU and leveraging large EP (≥16). This approach provides a scalable blueprint for high-performance MoE inference in GPU-rich environments, with near-linear scaling confirmed up to EP=64.

## 6. Key Technical Details
- **Communication**: Asynchronous token routing with NCCL
- **Load balancing**: Dynamic gating probability adjustment
- **Memory**: 33MB expert weights + 16GB activations per GPU
- **Topology**: 8×8 GPU grid with topology-aware routing
- **Bottleneck**: Network bandwidth at 75% utilization with overlap