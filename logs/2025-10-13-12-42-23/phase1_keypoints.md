# Phase 1: Key Points Extraction

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Technical Contributions

### 1. Expert Placement Strategy
- **Single-Expert-Per-GPU Deployment**: Deploy at most one expert per GPU to maximize expert-level parallelism
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, GPU memory, and routing patterns
- **EP ≥ 16**: Large Expert Parallelism regime with 16 or more experts per parallel group

### 2. Routing and Load Balancing
- **Gating Mechanism**: Top-K gating scores determine expert activation
- **Token Sharding**: Group tokens by destination expert, asynchronous routing, dynamic load balancing
- **Load Balancing**: Monitor per-expert load and adjust gating probabilities dynamically

### 3. Communication Overlap and Scheduling
- **Compute-Communication Overlap**: Interleave expert computation with cross-node token transfers
- **Pipeline Scheduling**: Fine-grained pipeline for multi-layer MoE networks
- **Asynchronous Communication**: Use CUDA streams/NCCL/MPI for non-blocking transfers

### 4. Scalability Considerations
- **Network Bandwidth Management**: Topology-aware routing and token batching
- **Memory Integration**: Tensor parallelism (TP) within GPUs when needed, Data parallelism (DP) across replicas
- **HPC Environment Optimization**: Designed for H100 clusters with high-bandwidth interconnects

## Experimental Results Summary
- **Model**: 4-layer MoE, 16 experts/layer, MLP experts
- **Precision**: FP16
- **Batch**: 1024 sequences × 10000 tokens
- **Performance Gain**: 3.75× higher throughput (450K vs 120K TPS), 3.8× lower latency (2.2ms vs 8.3ms TPOT)
- **Deployment**: 16 H100 GPUs, one expert per GPU vs baseline 8 experts per GPU

## Core Innovation
Shifting optimization focus from reducing communication to maximizing compute concurrency by fully utilizing distributed GPU resources with one-expert-per-GPU deployment.