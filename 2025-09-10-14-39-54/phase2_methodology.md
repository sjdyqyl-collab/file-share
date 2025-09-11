# Phase Two: Methodology

## 1. Overview
The method maximizes expert parallelism in MoE models by deploying **one expert per GPU** across nodes, focusing on three components: expert placement, routing/load balancing, and communication overlap.

## 2. Expert Placement Strategy
### 2.1 Single-Expert-Per-GPU Deployment
- For $E$ experts and $G$ GPUs: Assign 1 expert/GPU if $E ≤ G$; replicate experts if $E > G$ (balanced memory usage).
- Eliminates intra-GPU expert contention, fully utilizing compute resources.

### 2.2 Cross-Node Distribution
- Topology-aware placement considering node bandwidth/latency, GPU memory, and token routing patterns.
- Minimizes network hotspots by distributing experts across nodes.

## 3. Routing and Load Balancing
### 3.1 Gating Mechanism
- Standard MoE gating: Top-K experts activated per token.

### 3.2 Token Sharding
- **Batching**: Group tokens by destination expert to reduce network messages.
- **Asynchronous Routing**: Send batches without blocking computation (uses CUDA streams/NCCL).
- **Dynamic Gating**: Adjust gating probabilities to balance expert load and avoid stragglers.

## 4. Communication Overlap and Scheduling
### 4.1 Overlap Compute/Communication
- Interleave token transfer (cross-node) with GPU computation using asynchronous libraries (NCCL/MPI).
- Prevents GPU idle time during data transfer.

### 4.2 Pipeline Scheduling
- For multi-layer MoE: Process tokens in micro-batches through pipeline stages.
- Next layer’s experts start processing partial batches immediately (no full-batch wait).

## 5. Scalability Considerations
### 5.1 Large EP Regime (EP ≥ 16)
- Optimized for 16+ experts/layer: Network becomes the bottleneck, mitigated via topology-aware routing and batching.
- One-expert-per-GPU ensures full compute utilization.

### 5.2 Integration with Other Parallelism
- **Tensor Parallelism (TP)**: Apply TP to experts if they exceed GPU memory (e.g., TP=2 for large experts).
- **Data Parallelism (DP)**: Synchronize weights across MoE replicas for large models.

## 6. Advantages Summary
1. Maximal expert parallelism (no intra-GPU contention).
2. Balanced node load via topology-aware placement.
3. Scalable communication overlap (near-linear TPS scaling).
4. Compatibility with TP/DP for large models.