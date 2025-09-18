# Phase 1: Key Points Extraction

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Points

### 1. Problem Statement
- Traditional MoE parallelization strategies assign multiple experts to the same GPU to reduce inter-node communication
- This creates computational bottlenecks and limits true expert parallelism as models and clusters grow
- The trade-off becomes increasingly suboptimal with larger scales

### 2. Core Innovation
- Cross-node expert parallelism method that prioritizes distributing experts across nodes
- Each GPU hosts at most one expert
- Expert Parallelism (EP) pushed to 16 or beyond (defined as "large EP")
- Shifts optimization focus from reducing communication to maximizing compute concurrency

### 3. Technical Approach
- **Expert Placement**: One expert per GPU, distributed across nodes
- **Routing & Load Balancing**: Token batching, asynchronous routing, dynamic load balancing
- **Communication Overlap**: Interleave expert computation and communication using CUDA streams/NCCL
- **Pipeline Scheduling**: Fine-grained pipeline for multi-layer MoE networks

### 4. Scalability Features
- Optimized for EP ≥ 16 regime
- Network bandwidth becomes primary limiting factor (mitigated by topology-aware routing)
- Integrates with tensor model parallelism (TP) and data parallelism (DP) for large models
- Compatible with modern HPC networking (NVLink, InfiniBand, NVSwitch)

### 5. Experimental Validation
- **Model**: 4-layer MoE, 16 experts per layer, MLP experts
- **Precision**: FP16
- **Batch**: 1024 sequences, 10000 tokens per sequence
- **Token Dimension**: 8192
- **MHA**: 16 heads, 512 dimensions per head
- **MLP Hidden Size**: 32768
- **Metrics**: TPS (Tokens per Second), TPOT (Time per Output Token)

### 6. Results Summary
- **Baseline**: 16 H100 GPUs, TP=8, PP=2, 4 experts per GPU → 120,000 TPS, 8.3ms TPOT
- **Proposed**: 64 H100 GPUs, 1 expert per GPU → 450,000 TPS, 2.2ms TPOT
- **Improvement**: ~3.75× higher throughput, ~3.8× lower latency
- **Scaling**: Near-linear scaling with 64 GPUs in large EP regime