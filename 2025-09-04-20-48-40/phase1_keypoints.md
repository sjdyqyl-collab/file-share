# Phase 1: Keypoints Extraction

## Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Points

### Problem Statement
- Traditional MoE implementations colocate multiple experts on the same GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- As model and cluster sizes grow, this trade-off becomes suboptimal

### Proposed Solution
- **Large-scale cross-node expert parallelism**: Deploy at most one expert per GPU
- **Large EP definition**: EP ≥ 16 (experts per parallel group)
- **Core principle**: Shift bottleneck from intra-GPU contention to network communication

### Technical Innovations
1. **Single-Expert-Per-GPU Deployment**: Each GPU hosts exactly one expert when possible
2. **Cross-Node Distribution**: Topology-aware expert placement across nodes
3. **Asynchronous Token Routing**: Overlapping communication with computation
4. **Load Balancing**: Dynamic gating adjustments to prevent expert overload

### Performance Benefits
- **3.75× higher throughput** (450K vs 120K tokens/second)
- **3.8× lower latency** (2.2ms vs 8.3ms per token)
- **Linear scalability** in large EP regime (EP ≥ 16)
- **Full GPU utilization** without expert contention

### Deployment Configuration
- **Model**: 4-layer MoE, 16 experts per layer, MLP experts
- **Precision**: FP16
- **Batch size**: 1024 tokens per forward pass
- **Dimensions**: 16 attention heads, 512 head dimension, 32768 MLP hidden size
- **Hardware**: H100 GPUs
- **Baseline**: 16 GPUs (TP=8, PP=2, 4 experts per GPU)
- **Proposed**: 64 GPUs (1 expert per GPU, EP=64)