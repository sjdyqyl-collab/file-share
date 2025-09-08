# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction
Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, scaling MoE models across large GPU clusters introduces challenges in expert placement and parallelization. Traditional approaches colocate multiple experts on the same GPU to reduce communication, but this creates computational bottlenecks and limits expert parallelism.

We present a cross-node expert parallelism method that distributes experts such that each GPU hosts at most one expert, pushing EP to 16 or beyond. This design shifts optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities.

## Methods

### Overview
Our approach maximizes expert-level parallelism by deploying at most one expert per GPU and distributing experts across nodes. The method consists of three key components:

1. **Expert Placement Strategy** – Assigning experts across GPUs and nodes
2. **Routing and Load Balancing** – Ensuring balanced input distribution to experts
3. **Communication Overlap and Scheduling** – Minimizing cross-node data transfer impact

### Expert Placement Strategy

#### Single-Expert-Per-GPU Deployment
- Deploy at most one expert per GPU
- For E experts and G GPUs: assign each expert to distinct GPU if E ≤ G
- If E > G: replicate experts across GPUs to maximize concurrency
- Each expert processes tokens without contention from other experts

#### Cross-Node Distribution
- Topology-aware placement considering node-to-node bandwidth/latency, GPU memory capacity, and expected routing patterns
- Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU principle

### Routing and Load Balancing

#### Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities

### Communication Overlap and Scheduling

#### Overlapping Compute and Communication
- Interleave expert computation with token batch transfers
- Use CUDA streams or asynchronous communication libraries (NCCL/MPI)
- Ensure data transfer does not block GPU computation

#### Pipeline Scheduling
- Each MoE layer acts as a micro-stage
- Token outputs immediately routed to next layer's experts
- Experts start processing as soon as partial batch arrives

### Scalability Considerations

#### Large EP Regime (EP ≥ 16)
- Network bandwidth becomes primary limiting factor
- Mitigated through topology-aware routing and token batching
- One-expert-per-GPU ensures full GPU utilization

#### Memory and Model Parallelism Integration
- Tensor Parallelism (TP): Applied within expert if FFN cannot fit on single GPU
- Data Parallelism (DP): Applied across MoE network replicas

## Experiments

### Experimental Setup
- **Model**: 4-layer MoE, 16 experts per layer, each expert is an MLP
- **Precision**: FP16
- **Batch size**: 1024 sequences
- **Sequence Length**: 10,000 tokens per sequence
- **Token Dimension**: 8,192
- **MHA**: 16 heads × 512 = 8,192 dimensions
- **MLP Hidden Size**: 32,768

### Parallel Deployment Details

#### Baseline Deployment (TP=8, PP=2)
- **GPUs**: 16 H100
- **Configuration**: 4 experts + TP shard per GPU, 2 pipeline stages
- **Limitation**: GPUs shared among experts causing intra-GPU contention

#### Proposed Cross-Node Expert Parallelism
- **GPUs**: 64 H100 (one GPU per expert per layer)
- **Configuration**: 1 expert per GPU, each MoE layer as micro-stage
- **Advantage**: All 64 experts per layer compute in parallel

### Results
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert per GPU | 450,000 | 2.2ms |

### Performance Improvements
- **Throughput**: 3.75× higher (450,000 vs 120,000 TPS)
- **Latency**: 3.8× lower (2.2ms vs 8.3ms TPOT)
- **Scalability**: Near-linear scaling in large EP regime

## Conclusion
We proposed a large-scale cross-node expert parallelism method that maximizes expert-level parallelism by deploying at most one expert per GPU. Our approach achieved 3.75× higher throughput and 3.8× lower latency compared to baseline, validating that distributing experts across GPUs and overlapping communication with computation dramatically improves performance for large-scale MoE deployments. This provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.