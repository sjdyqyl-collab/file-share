# Phase 1: Keypoints Extraction

## Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Contributions

### 1. Large Expert Parallelism (EP ≥ 16)
- Definition: Expert Parallelism degree of 16 or more
- Each GPU hosts at most one expert
- Maximizes compute concurrency over communication optimization

### 2. Expert Placement Strategy
- Single-expert-per-GPU deployment principle
- Cross-node distribution with topology awareness
- Memory capacity and bandwidth considerations

### 3. Routing and Load Balancing
- Token batching by destination expert
- Asynchronous routing to overlap communication
- Dynamic load balancing through gating probability adjustment

### 4. Communication Overlap
- Interleaved compute and communication
- CUDA streams/NCCL for asynchronous operations
- Pipeline scheduling across MoE layers

## Core Methodology

### Model Architecture
- 4-layer MoE model
- 16 experts per layer
- Each expert: MLP with hidden size 32768
- MHA: 16 heads, 512 dimensions per head
- Precision: FP16
- Batch size: 1024 tokens

### Parallel Strategies
- **Proposed**: EP=64 (one expert per GPU), optional TP=2 per expert
- **Baseline**: TP=8, PP=2 with 4 experts per GPU

## Performance Results

| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert per GPU | 450,000 | 2.2ms |

**Improvement**: 3.75× higher throughput, 3.8× lower latency

## Technical Specifications
- **Hardware**: H100 GPUs
- **Network**: NVLink, InfiniBand, H100-class NVSwitch
- **Libraries**: NCCL, MPI for communication
- **Setting**: Inference-only deployment