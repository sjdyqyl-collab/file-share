# Methodology - Large-Scale Cross-Node Expert Parallelism

## Overview
Our approach focuses on maximizing expert-level parallelism in large-scale Mixture-of-Experts (MoE) models by deploying at most one expert per GPU, and distributing experts across nodes to exploit available compute resources fully. The core idea is to shift the bottleneck from inter-expert contention to network communication, which can be mitigated through careful scheduling, routing, and overlapping of communication and computation.

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Rule**: Deploy at most one expert per GPU
- **Implementation**: 
  - For E experts and G GPUs: assign each expert to distinct GPU if E ≤ G
  - If E > G: replicate experts across GPUs to maximize concurrency
- **Benefit**: Each expert processes tokens without contention from other experts

### 1.2 Cross-Node Distribution
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU principle

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard MoE gating network determines expert activation via top-K gating scores

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Interleaving**: Process one batch while transferring next batch
- **Implementation**: CUDA streams or asynchronous communication libraries (NCCL/MPI)
- **Non-blocking**: Data transfer does not block GPU computation

### 3.2 Pipeline Scheduling
- **Micro-staging**: Each MoE layer as a micro-stage
- **Immediate routing**: Token outputs immediately routed to next layer's experts
- **Partial processing**: Experts start processing as soon as partial batch arrives

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Network bandwidth**: Primary limiting factor
- **Mitigation**: Topology-aware routing + token batching
- **One-expert-per-GPU**: Ensures full GPU utilization

### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Applied within expert if FFN cannot fit on single GPU
- **Data Parallelism (DP)**: Applied across MoE network replicas
- **Synchronized updates**: Maintains high expert-level parallelism

## 5. Deployment Configuration Summary
- **Parallelism degrees**: 
  - Expert Parallelism (EP) = 16 (minimum for large EP)
  - Tensor Parallelism (TP) = optional (2 if needed)
  - Pipeline Parallelism (PP) = micro-stages per layer
- **GPU allocation**: One GPU per expert
- **Communication**: Asynchronous token routing with overlap
- **Load balancing**: Dynamic gating probability adjustment