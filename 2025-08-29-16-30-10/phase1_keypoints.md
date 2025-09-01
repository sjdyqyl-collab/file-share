# Phase 1: Key Points Extraction

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Technical Contributions

### 1. Core Innovation: Single-Expert-Per-GPU Deployment
- **Principle**: Deploy at most one expert per GPU to maximize expert-level parallelism
- **Definition**: Large EP = EP ≥ 16 (at least 16 experts per parallel group)
- **Goal**: Shift bottleneck from intra-GPU contention to network communication

### 2. Three Key Components
1. **Expert Placement Strategy**: Assigning experts across GPUs and nodes
2. **Routing and Load Balancing**: Ensuring balanced input distribution to experts
3. **Communication Overlap and Scheduling**: Minimizing cross-node data transfer impact

### 3. Expert Placement Strategy Details
- **Single-Expert-Per-GPU**: Each expert assigned to distinct GPU if E ≤ G
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, memory capacity
- **Memory Integration**: Tensor parallelism within GPU if expert exceeds memory

### 4. Routing and Load Balancing
- **Gating Mechanism**: Top-K gating scores determine expert activation
- **Token Sharding**: Group tokens by destination expert, asynchronous routing, dynamic load balancing

### 5. Communication Optimization
- **Overlapping Compute and Communication**: Interleave expert computation with token transfers
- **Pipeline Scheduling**: Immediate routing between layers, partial batch processing

## Experimental Results Summary

### Model Configuration
- **Architecture**: 4-layer MoE, 16 experts per layer, MLP experts
- **Precision**: FP16
- **Batch Size**: 1024 tokens per forward pass
- **Dimensions**: 16 attention heads × 512 dimensions, MLP hidden size 32768

### Performance Comparison
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert per GPU | 450,000 | 2.2ms |

### Key Achievements
- **3.75× higher throughput** (450k vs 120k TPS)
- **3.8× lower latency** (2.2ms vs 8.3ms TPOT)
- **Near-linear scaling** with 64 GPUs in large EP regime