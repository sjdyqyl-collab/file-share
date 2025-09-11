# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models - Concise Version

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## 1. Introduction
Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, scaling MoE models across large GPU clusters introduces significant challenges in expert placement and parallelization. Traditional approaches assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks that limit expert-level parallelism. Our method prioritizes distributing experts across nodes with at most one expert per GPU, pushing EP to 16 or beyond to unlock higher degrees of concurrent computation.

## 2. Methods

### 2.1 Expert Placement Strategy
**Single-Expert-Per-GPU Principle**: Each GPU hosts at most one expert, ensuring minimal contention and maximum compute utilization. For E experts and G GPUs, each expert is assigned to a distinct GPU if E ≤ G. If E > G, experts are replicated to maximize concurrent independent experts.

**Cross-Node Distribution**: Topology-aware placement considering node-to-node bandwidth, GPU memory capacity, and token routing patterns to minimize hotspotting and balance load.

### 2.2 Routing and Load Balancing
**Token Sharding Protocol**:
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously while overlapping expert computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities (threshold: |L[e] - mean(L)|/mean(L) < 0.2)

### 2.3 Communication Overlap and Scheduling
**Overlapping Strategy**: Interleave expert computation and communication using CUDA streams and NCCL/MPI. While batch n is processed, transfer batch n+1 simultaneously with ≥80% overlap ratio.

**Pipeline Scheduling**: Each MoE layer is a micro-stage with immediate routing between layers and partial batch processing to minimize idle time.

### 2.4 Scalability Framework
**Large EP Regime (EP ≥ 16)**: Optimized for configurations with 16+ experts per parallel group where network bandwidth becomes the primary limiting factor.

**Hybrid Parallelism Integration**: Compatible with tensor model parallelism (TP=2 optional within expert) and data parallelism (DP) for models exceeding single-GPU memory.

## 3. Experimental Setup

### 3.1 Model Configuration
- **Architecture**: 4-layer MoE with 16 experts per layer
- **Expert Type**: MLP with hidden size 32,768
- **Precision**: FP16
- **Token Dimension**: 8,192
- **MHA**: 16 heads × 512 dimensions = 8,192 total

### 3.2 Input Configuration
- **Batch Size**: 1,024 sequences
- **Sequence Length**: 10,000 tokens
- **Total Tokens**: 10,240,000 per batch

### 3.3 Hardware Setup
- **GPUs**: H100 cluster
- **Baseline**: 16 GPUs (TP=8, PP=2) with 4 experts per GPU
- **Proposed**: 64 GPUs with 1 expert per GPU

## 4. Results

| Method | GPUs | Deployment | TPS | TPOT (ms) | Improvement |
|--------|------|------------|-----|-----------|-------------|
| Baseline | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 | - |
| Proposed | 64 | 1 expert per GPU | 450,000 | 2.2 | 3.75× TPS, 3.8× latency |

**Key Achievements**:
- **3.75× higher throughput** (450,000 vs 120,000 TPS)
- **3.8× lower latency** (2.2ms vs 8.3ms TPOT)
- **Linear scaling** with 93.75% efficiency (3.75× with 4× GPUs)
- **Full GPU utilization** with dedicated expert computation

## 5. Technical Implementation Details

### 5.1 Memory Requirements
- **Per Expert**: ~2.3 GB (MLP parameters + overhead)
- **Baseline**: 9.2 GB per GPU (4 experts shared)
- **Proposed**: 2.3 GB per GPU (1 expert dedicated)

### 5.2 Communication Patterns
- **Token Transfer**: 16,384 bytes per token (8,192 × FP16)
- **Per Batch**: 167.8 GB total communication
- **Required Bandwidth**: 473 GB/s cluster-wide for 450,000 TPS
- **Interconnect**: NVLink (600 GB/s), InfiniBand (400 Gbps)

### 5.3 Runtime Configuration
- **Framework**: Custom MoE implementation
- **Libraries**: NCCL 2.18+, CUDA 12.x, MPI
- **Optimizations**: Async communication, pipeline scheduling, load balancing
- **Load Balancing Threshold**: 20% deviation from mean expert load

## 6. Conclusion
Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU. By pushing EP to 16+ and leveraging modern HPC networking, we achieve 3.75× higher throughput and 3.8× lower latency compared to traditional approaches. This provides a scalable blueprint for high-performance MoE inference in GPU-rich environments, with potential extensions to training scenarios and even larger expert counts.

## 7. Deployment Configuration Summary

### 7.1 Baseline Model (16 GPUs)
- **Parallelism**: TP=8, PP=2, EP=2
- **Expert Placement**: 4 experts per GPU
- **Communication**: Tensor parallelism within stages, pipeline between stages

### 7.2 Proposed Model (64 GPUs)
- **Parallelism**: EP=64, PP=4, TP=1 (within expert)
- **Expert Placement**: 1 expert per GPU across 4 nodes
- **Communication**: All-to-all for token routing, pipeline between layers
- **Large EP Regime**: EP=64 ≥ 16 threshold

The method successfully shifts the optimization focus from reducing communication to maximizing compute concurrency, demonstrating the viability of large-scale expert parallelism for future MoE deployments.