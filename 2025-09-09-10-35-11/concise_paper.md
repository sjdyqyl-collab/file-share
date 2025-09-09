# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, traditional MoE parallelization strategies colocate multiple experts on the same GPU to reduce inter-node communication, creating computational bottlenecks that limit expert-level parallelism as model and cluster sizes grow.

We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This design shifts optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities.

## Methods

### Core Design Principles

**Large Expert Parallelism (EP ≥ 16)**: Deploy at most one expert per GPU to eliminate intra-GPU contention and maximize parallel computation.

**Cross-Node Distribution**: Topology-aware expert placement across nodes considering bandwidth, latency, and memory capacity.

### Expert Placement Strategy

**Single-Expert-Per-GPU Deployment**:
- Each GPU hosts exactly one expert
- No expert parameter sharing within GPU
- Tensor parallelism applied only if single expert exceeds GPU memory

**Cross-Node Distribution Algorithm**:
1. Calculate communication matrix based on expected expert activation patterns
2. Apply graph partitioning to minimize cross-node traffic
3. Ensure load balancing across all GPUs
4. Validate memory constraints per node

### Routing and Load Balancing

**Token Sharding Strategy**:
- Group tokens by destination expert to reduce network messages
- Send token batches asynchronously to overlap with computation
- Monitor per-expert load and dynamically adjust gating probabilities

**Dynamic Load Balancing**:
- Real-time monitoring of expert utilization
- Adjustment of gating probabilities to prevent overloading
- Token dropping threshold for severely overloaded experts

### Communication Overlap and Scheduling

**CUDA Stream Architecture**:
- **Compute Stream**: Handles expert computation
- **Communication Stream**: Handles token transfers
- **Synchronization**: CUDA events for stream coordination

**Pipeline Scheduling**:
- Each MoE layer treated as micro-stage
- Token outputs immediately routed to next layer's experts
- Fine-grained pipeline with token-level granularity

## Experiments

### Experimental Setup

**Model Configuration**:
- **Architecture**: 4-layer MoE with 16 experts per layer
- **Expert Type**: Multi-Layer Perceptron (MLP)
- **Precision**: FP16
- **Input**: 1024 sequences × 10,000 tokens × 8,192 dimensions

**Hardware**: H100 GPUs with NVLink 4.0 and InfiniBand HDR

### Baseline vs Proposed Comparison

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### Results Analysis

**Performance Improvements**:
- **3.75× higher throughput** (450K vs 120K tokens/second)
- **3.8× lower latency** (2.2ms vs 8.3ms per token)
- **94% scaling efficiency** from 16 to 64 GPUs

**Key Insights**:
- One-expert-per-GPU eliminates intra-GPU contention
- Asynchronous token routing effectively hides network latency
- Large EP ≥ 16 enables near-linear scaling in HPC environments

## Conclusion

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU, achieving 3.75× higher throughput and 3.8× lower latency compared to traditional approaches. The paradigm shift from minimizing communication to maximizing compute concurrency proves effective for large-scale MoE deployments in HPC environments with abundant GPU resources.

## Deployment Configuration Summary

**Requirements**:
- **GPUs**: 64 H100 GPUs (80GB each)
- **Network**: 450 GB/s aggregate bandwidth
- **Parallelism**: EP=64, micro-stage pipeline
- **Memory**: 3GB per GPU (537MB expert + 2.5GB buffer)

**Key Parameters**:
- Expert size: 8192 × 32768 parameters
- Token dimension: 8,192
- Batch size: 1024 sequences × 10,000 tokens
- Precision: FP16