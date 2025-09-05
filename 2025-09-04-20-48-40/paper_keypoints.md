# Key Points Extraction - Large-Scale Cross-Node Expert Parallelism for MoE Models

## Abstract (Original)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Innovations

### 1. Large Expert Parallelism (EP ≥ 16)
- **Definition**: Configurations with 16 or more experts per parallel group
- **Key Principle**: One expert per GPU maximum
- **Benefit**: Minimizes resource contention and maximizes expert-level parallel execution

### 2. Expert Placement Strategy
- **Single-Expert-Per-GPU Deployment**: Each GPU hosts at most one expert
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, GPU memory, and routing patterns
- **Scalability**: When E > G, experts are replicated across GPUs to maximize concurrency

### 3. Routing and Load Balancing
- **Gating Mechanism**: Top-K gating scores determine expert activation
- **Token Sharding**: Efficient cross-node token transfer through batching and asynchronous routing
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities

### 4. Communication Overlap and Scheduling
- **Compute-Communication Overlap**: Interleave expert computation with token transfers
- **Pipeline Scheduling**: Token outputs immediately routed to next layer experts
- **Asynchronous Operations**: CUDA streams/NCCL for non-blocking data transfer

## Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts**: 16 experts per layer, each expert is an MLP
- **Precision**: FP16
- **Batch Size**: 1024 tokens per forward pass
- **Multi-Head Attention**: 16 heads, 512 dimensions per head
- **MLP Hidden Size**: 32768

## Performance Metrics
- **TPS (Tokens per Second)**: Throughput measurement
- **TPOT (Time per Output Token)**: Latency per token measurement

## Deployment Comparison

### Baseline Configuration (TP=8, PP=2)
- **GPUs**: 16 H100
- **Per-GPU Allocation**: 4 experts + TP shard per GPU
- **Pipeline**: 2 stages, 8 GPUs each
- **Results**: 120,000 TPS, 8.3ms TPOT

### Proposed Method
- **GPUs**: 64 H100
- **Per-GPU Allocation**: 1 expert per GPU
- **Parallelism**: EP=64 (one expert per GPU per layer)
- **Results**: 450,000 TPS, 2.2ms TPOT
- **Improvement**: 3.75× higher throughput, 3.8× lower latency

## Critical Dimensions and Parameters
- **Expert Count**: 64 experts per layer (4 layers × 16 experts)
- **GPU Count**: 64 H100 GPUs (one per expert)
- **Tensor Parallelism**: Optional TP=2 if expert FFN exceeds GPU memory
- **Communication**: Asynchronous token routing with NCCL/MPI
- **Memory**: Each GPU hosts exactly one expert (MLP with hidden size 32768)
- **Batch Processing**: 1024 tokens distributed across 64 experts