# Phase 1: Key Points Extraction

## Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Points

### 1. Problem Statement
- Traditional MoE implementations colocate multiple experts on same GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- As models and clusters grow, this trade-off becomes suboptimal

### 2. Core Innovation
- **Large Expert Parallelism (EP ≥ 16)**: Distribute experts across nodes with at most one expert per GPU
- Maximizes compute concurrency by shifting bottleneck from contention to network communication
- Leverages modern HPC networking capabilities (NVLink, InfiniBand, NVSwitch)

### 3. Technical Approach
- **Expert Placement**: One expert per GPU, topology-aware distribution across nodes
- **Routing**: Dynamic token routing with load balancing and asynchronous communication
- **Overlap**: Interleave computation and communication using CUDA streams/NCCL

### 4. Experimental Results
- **Model**: 4-layer MoE, 16 experts per layer, MLP experts
- **Precision**: FP16
- **Batch**: 1024 sequences × 10000 tokens
- **Dimensions**: Token dim 8192, MHA heads 16×512, MLP hidden 32768
- **Results**: 3.75× higher throughput (450k vs 120k TPS), 3.8× lower latency (2.2ms vs 8.3ms TPOT)

### 5. Deployment Comparison
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 H100 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed | 64 H100 | 1 expert per GPU | 450,000 | 2.2ms |

### 6. Scalability Advantages
- **Maximized Expert Parallelism**: One expert per GPU ensures minimal contention
- **Balanced Load**: Topology-aware placement prevents network bottlenecks
- **Communication Overlap**: Asynchronous routing enables near-linear scaling
- **Model Integration**: Compatible with TP and DP for large models