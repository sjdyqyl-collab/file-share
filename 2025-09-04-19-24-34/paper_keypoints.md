# Key Points of Large-Scale Cross-Node Expert Parallelism for MoE Models

## Core Problem
Traditional MoE parallelization assigns multiple experts per GPU to reduce communication, but creates computational bottlenecks and limits expert-level parallelism.

## Proposed Solution
Large-scale cross-node expert parallelism with at most one expert per GPU, pushing Expert Parallelism (EP) to ≥16.

## Key Innovations
1. **Single-expert-per-GPU deployment** - Eliminates intra-GPU contention
2. **Cross-node distribution** - Topology-aware expert placement across nodes
3. **Asynchronous token routing** - Overlaps communication with computation
4. **Large EP regime** - EP ≥ 16 for maximum expert independence

## Technical Components
- **Expert Placement**: One expert per GPU when E ≤ G, replicated experts when E > G
- **Routing**: Top-K gating with token batching and load balancing
- **Communication Overlap**: CUDA streams/NCCL for async communication
- **Scalability**: Integrates with TP and DP for memory-constrained models

## Experimental Validation
- Model: 4-layer MoE, 16 experts/layer, 32768 hidden size MLP
- Precision: FP16, Batch: 1024 tokens
- Results: 3.75× higher TPS (450k vs 120k), 3.8× lower latency (2.2ms vs 8.3ms)
- Setup: 64 H100 GPUs (1 expert/GPU) vs 16 H100 baseline (4 experts/GPU)

## Deployment Impact
- Maximizes compute concurrency over communication optimization
- Near-linear scaling in HPC environments with high-bandwidth interconnects
- Particularly effective in H100-class clusters with NVSwitch/InfiniBand

## Original Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.