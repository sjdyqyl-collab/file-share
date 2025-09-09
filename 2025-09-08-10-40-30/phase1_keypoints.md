# Phase 1: Keypoints Extraction

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Keypoints

### 1. Core Problem
- Traditional MoE parallelization assigns multiple experts per GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- As model and cluster sizes grow, this trade-off becomes suboptimal

### 2. Proposed Solution
- **Large-scale cross-node expert parallelism** with at most one expert per GPU
- **Large EP regime**: EP ≥ 16 (16 or more experts per parallel group)
- Prioritize distributing experts across nodes to maximize compute concurrency
- Shift optimization focus from reducing communication to maximizing compute concurrency

### 3. Key Components
1. **Expert Placement Strategy**: Assign at most one expert per GPU
2. **Routing and Load Balancing**: Ensure balanced input distribution to experts
3. **Communication Overlap and Scheduling**: Minimize cross-node data transfer impact

### 4. Technical Innovations
- **Single-expert-per-GPU deployment**: Each expert runs in isolation
- **Cross-node distribution**: Topology-aware placement considering bandwidth, latency, memory
- **Token batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous routing**: Send token batches asynchronously to overlap with computation
- **Pipeline scheduling**: Fine-grained pipeline to increase throughput

### 5. Experimental Results
- **Model**: 4-layer MoE, 16 experts per layer, FP16 precision
- **Baseline**: TP=8, PP=2, 16 GPUs, 4 experts per GPU → 120,000 TPS, 8.3ms TPOT
- **Proposed**: 64 GPUs, 1 expert per GPU → 450,000 TPS, 2.2ms TPOT
- **Improvement**: ~3.75× higher throughput, ~3.8× lower latency

### 6. Advantages
1. Maximized Expert Parallelism: One expert per GPU ensures minimal contention
2. Balanced Load Across Nodes: Topology-aware placement prevents bottlenecks
3. Scalable Communication Overlap: Asynchronous routing enables near-linear scaling
4. Compatibility with Large Models: Integrates with TP and DP for models exceeding single-GPU memory