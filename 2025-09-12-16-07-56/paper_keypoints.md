# Phase One: Keypoints Extraction

## Abstract (Retained Original)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Problem & Motivation
- Traditional MoE parallelization assigns multiple experts to same GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- Network technology advances (NVLink, InfiniBand, NVSwitch) make communication less dominant
- Need to shift focus from reducing communication to maximizing compute concurrency

## Core Innovation: Large Expert Parallelism (EP ≥ 16)
- Definition: Large EP = configurations with 16 or more experts per parallel group
- Key principle: Deploy at most ONE expert per GPU
- Goal: Maximize expert-level parallelism and minimize contention
- Trade-off: Accept higher communication cost for better compute utilization

## Technical Components

### 1. Expert Placement Strategy
- Single-expert-per-GPU deployment when E ≤ G (experts ≤ GPUs)
- If E > G: replicate experts to maximize concurrency while balancing memory
- Topology-aware placement considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns

### 2. Routing and Load Balancing
- Standard MoE gating mechanism with top-K expert selection
- Token sharding across nodes:
  - Token batching by destination expert
  - Asynchronous routing to overlap with computation
  - Dynamic load balancing to prevent expert overload

### 3. Communication Overlap and Scheduling
- Interleave expert computation and communication
- Use CUDA streams/NCCL/MPI for asynchronous transfers
- Pipeline scheduling for multi-layer MoE networks
- Immediate routing between layers without waiting for full batch

## Experimental Configuration
- Model: 4-layer MoE, 16 experts per layer (64 experts total)
- Precision: FP16
- Batch: 1024 sequences × 10000 tokens = 10.24M tokens
- Token dimension: 8192
- MHA: 16 heads × 512 dim per head
- MLP hidden size: 32768
- Metrics: TPS (Tokens/second), TPOT (ms/token)

## Results Summary
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert per GPU | 450,000 | 2.2ms |

**Performance Gains:**
- 3.75× higher throughput
- 3.8× lower latency
- Near-linear scaling with 64 GPUs

## Critical Dimensions & Parameters
- Expert count: 16 per layer × 4 layers = 64 total
- Token dimension: 8192
- MLP hidden dimension: 32768
- Batch size: 1024 sequences
- Sequence length: 10000 tokens
- Precision: FP16
- GPU type: H100

## Deployment Requirements
- Minimum EP degree: 16
- Preferred: One GPU per expert
- Network: High-bandwidth interconnects (NVLink, InfiniBand)
- Memory: Sufficient for single expert per GPU
- Software: NCCL/MPI for communication, CUDA streams for overlap