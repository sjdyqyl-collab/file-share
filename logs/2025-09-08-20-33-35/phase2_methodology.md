# Phase 2: Methodology Extraction

## Overview
Our approach maximizes expert-level parallelism in large-scale MoE models by deploying at most one expert per GPU, distributing experts across nodes to fully exploit available compute resources. The core idea shifts optimization from reducing communication to maximizing compute concurrency.

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: At most one expert per GPU
- **Implementation**: 
  - For E experts and G GPUs: assign each expert to distinct GPU if E ≤ G
  - If E > G: replicate experts across GPUs to maximize concurrency while balancing memory
- **Benefit**: Each expert processes tokens without contention from other experts on same device

### 1.2 Cross-Node Distribution
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard MoE routing: top-K gating scores determine expert activation per token

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously while overlapping expert computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities to prevent overloading

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Interleaving Strategy**:
  - While one batch processes on GPU, next batch transfers from other nodes
  - CUDA streams or NCCL/MPI for asynchronous communication
- **Implementation**: Data transfer does not block GPU computation

### 3.2 Pipeline Scheduling
- **Multi-layer coordination**:
  - Token outputs immediately routed to next layer's experts
  - Subsequent layer experts start processing partial batches without waiting for full batch
- **Benefit**: Fine-grained pipeline increases throughput, reduces idle time

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Expert Parallelism degree ≥ 16
- **Characteristics**:
  - Network bandwidth becomes primary limiting factor
  - One-expert-per-GPU ensures full GPU utilization
  - Communication costs amortized across many tokens

### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Applied within single expert's FFN if cannot fit on one GPU
- **Data Parallelism (DP)**: Applied across MoE network replicas for synchronized weight updates
- **Compatibility**: Seamless integration with TP and DP for models exceeding single-GPU memory

## 5. Technical Parameters and Dimensions
- **Model Architecture**: 4-layer MoE transformer
- **Experts per Layer**: 16 (total 64 experts across 4 layers)
- **Token Dimension**: 8192
- **MHA Configuration**: 16 heads × 512 dimensions per head
- **MLP Hidden Size**: 32768
- **Precision**: FP16
- **Batch Configuration**: 1024 sequences × 10000 tokens per sequence
- **Hardware**: H100 GPUs

## 6. Method Summary
The methodology creates a three-tier optimization:
1. **Placement**: One expert per GPU with topology-aware distribution
2. **Routing**: Dynamic, asynchronous token routing with load balancing
3. **Scheduling**: Overlapping computation and communication through pipelining

This approach fundamentally rethinks MoE parallelization by prioritizing compute concurrency over communication minimization, enabled by modern HPC networking capabilities.