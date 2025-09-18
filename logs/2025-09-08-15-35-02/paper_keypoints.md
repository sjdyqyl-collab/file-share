# Key Points of the Paper: Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Core Problem and Motivation
- Traditional MoE implementations colocate multiple experts on the same GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- With modern HPC networking (NVLink, InfiniBand, NVSwitch), communication cost is less dominant than compute concurrency gains

## Key Innovation: Large Expert Parallelism (EP ≥ 16)
- Deploy at most one expert per GPU
- Distribute experts across nodes to maximize computational parallelism
- Shift bottleneck from inter-expert contention to network communication
- Define "large EP" as configurations with EP ≥ 16

## Methodology Components

### 1. Expert Placement Strategy
- Single-expert-per-GPU deployment
- If E ≤ G: each expert assigned to distinct GPU
- If E > G: replicate experts to maximize concurrency while balancing memory
- Cross-node distribution with topology-aware placement considering bandwidth, latency, memory capacity, and routing patterns

### 2. Routing and Load Balancing
- Gating mechanism determines expert activation (top-K scores)
- Token sharding across nodes with:
  - Token batching by destination expert
  - Asynchronous routing
  - Dynamic load balancing by adjusting gating probabilities

### 3. Communication Overlap and Scheduling
- Interleave expert computation and communication
- Use CUDA streams/NCCL/MPI for asynchronous communication
- Pipeline scheduling for multi-layer MoE networks
- Token outputs immediately routed to next layer's experts

## Experimental Configuration
- Model: 4-layer MoE, 16 experts per layer, each expert is MLP
- Precision: FP16
- Batch: 1024 sequences, 10000 tokens per sequence
- Token dimension: 8192
- MHA: 16 heads, 512 dim per head
- MLP hidden size: 32768
- Inference-only setting on H100 GPUs

## Baseline vs Proposed Deployment

### Baseline (TP=8, PP=2)
- 16 H100 GPUs
- 4 experts + TP shard per GPU
- Sequential pipeline processing
- Multiple experts share GPU compute

### Proposed Method
- 64 H100 GPUs (one GPU per expert per layer)
- Exactly one expert per GPU
- Optional TP=2 only if single expert FFN cannot fit
- Dynamic routing with asynchronous token batches
- All 64 experts per layer compute in parallel

## Key Results
- Throughput: 450,000 tokens/s (vs 120,000 baseline) - 3.75× improvement
- Latency: 2.2ms per token (vs 8.3ms baseline) - 3.8× improvement
- Near-linear scaling in large EP regime (EP ≥ 16)

## Scalability Advantages
1. Maximized Expert Parallelism: Minimal contention, high compute efficiency
2. Balanced Load: Topology-aware placement prevents bottlenecks
3. Communication Overlap: Asynchronous routing enables near-linear scaling
4. Large Model Compatibility: Integrates with TP and DP for memory constraints

## Technical Specifications for Deployment
- Expert Parallelism: EP = 64 (16 experts × 4 layers)
- Tensor Parallelism: TP = 1 (baseline TP=8, optional TP=2 in proposed)
- Pipeline Parallelism: PP = 2 (baseline), PP = 1 per layer (proposed)
- Cross-node communication with token batching and overlap
- Memory requirement: One expert per GPU with 32768 hidden size