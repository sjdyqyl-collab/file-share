# Phase 1: Keypoints Extraction

## Abstract (Retained as-is)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Points

### Problem Statement
- Traditional MoE parallelization assigns multiple experts per GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- As model/cluster sizes grow, this becomes increasingly suboptimal

### Proposed Solution
- **Large-scale cross-node expert parallelism** with EP ≥ 16
- Deploy **at most one expert per GPU** to maximize compute concurrency
- Shift optimization focus from reducing communication to maximizing compute parallelism
- Leverage modern HPC networking (NVLink, InfiniBand, NVSwitch) to handle communication overhead

### Core Components
1. **Expert Placement Strategy**: One-expert-per-GPU deployment, topology-aware distribution
2. **Routing and Load Balancing**: Dynamic gating with token batching and asynchronous routing
3. **Communication Overlap**: Interleave computation and communication using CUDA streams/NCCL
4. **Pipeline Scheduling**: Fine-grained pipeline for multi-layer MoE networks

### Technical Specifications
- **Model**: 4-layer MoE with 16 experts per layer (64 total experts)
- **Precision**: FP16
- **Batch Size**: 1024 sequences × 10000 tokens per sequence
- **Architecture**: MHA with 16 heads × 512 dimensions, MLP hidden size 32768
- **Large EP Regime**: EP ≥ 16

### Performance Results
- **Baseline (TP=8, PP=2)**: 16 GPUs, 4 experts per GPU → 120,000 TPS, 8.3ms TPOT
- **Proposed Method**: 64 GPUs, 1 expert per GPU → 450,000 TPS, 2.2ms TPOT
- **Improvement**: 3.75× higher throughput, 3.8× lower latency

### Key Innovations
- **Single-expert-per-GPU**: Eliminates intra-GPU contention
- **Cross-node distribution**: Maximizes expert-level parallelism
- **Asynchronous token routing**: Overlaps communication with computation
- **Near-linear scaling**: Demonstrated for EP ≥ 16 regime
- **Integration ready**: Compatible with TP and DP for memory-constrained scenarios

### Deployment Context
- **Environment**: H100 GPU clusters
- **Setting**: Inference-only (with potential for training extension)
- **Scalability**: Designed for environments with abundant GPU resources
- **Future work**: Training scenarios, dynamic routing, larger expert counts