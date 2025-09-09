# Phase 1: Key Points Extraction

## Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Points

### Core Problem
- Traditional MoE parallelization assigns multiple experts to the same GPU, creating computational bottlenecks
- This limits the degree of true expert parallelism as models and clusters grow

### Proposed Solution
- **Large-scale cross-node expert parallelism** with at most one expert per GPU
- **Large EP regime**: Expert Parallelism ≥ 16
- Focus shifts from reducing communication to maximizing compute concurrency

### Key Innovations
1. **Single-expert-per-GPU deployment** - eliminates intra-GPU contention
2. **Cross-node distribution** - topology-aware placement across nodes
3. **Asynchronous token routing** - overlap communication with computation
4. **Dynamic load balancing** - prevent expert overloading

### Technical Approach
- **Expert Placement Strategy**: One expert per GPU, distributed across nodes
- **Routing & Load Balancing**: Token batching, asynchronous routing, dynamic gating
- **Communication Overlap**: CUDA streams/NCCL for async communication
- **Pipeline Scheduling**: Micro-stages for immediate token routing

### Model Specifications
- **Architecture**: 4-layer MoE with 16 experts per layer
- **Expert Type**: MLP-based experts
- **Precision**: FP16
- **Token Dimension**: 8192
- **Sequence Length**: 10,000 tokens
- **Batch Size**: 1024 sequences
- **MLP Hidden Size**: 32,768
- **MHA**: 16 heads, 512 dimensions per head

### Performance Results
- **Baseline (TP=8, PP=2)**: 120,000 TPS, 8.3ms TPOT with 16 GPUs
- **Proposed Method**: 450,000 TPS, 2.2ms TPOT with 64 GPUs
- **Improvement**: 3.75× higher throughput, 3.8× lower latency

### Deployment Configurations
- **Baseline**: 16 H100 GPUs, 4 experts per GPU, TP=8, PP=2
- **Proposed**: 64 H100 GPUs, 1 expert per GPU, EP=64, optional TP=2 per expert

### Scalability Features
- Compatible with tensor model parallelism (TP) within each expert
- Integrates with data parallelism (DP) across MoE replicas
- Near-linear scaling in large EP regime (EP ≥ 16)