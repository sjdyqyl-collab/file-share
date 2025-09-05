# Phase 2: Methodology Extraction

## Overview of Proposed Method
The method maximizes expert-level parallelism in large-scale MoE models by deploying at most one expert per GPU, distributing experts across nodes to fully exploit available compute resources. This shifts the bottleneck from intra-GPU contention to network communication, which is mitigated through careful scheduling, routing, and overlapping of communication and computation.

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: Each GPU hosts at most one expert
- **Mathematical formulation**: For MoE layer with E experts and cluster of G GPUs:
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Replicate experts across GPUs while maximizing concurrency of independent experts
- **Benefit**: Each expert processes tokens without contention from other experts on same device

### 1.2 Cross-Node Distribution
- **Topology-aware placement strategy** considers:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum number of tokens sent across any single link while maintaining one-expert-per-GPU principle

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard MoE gating network determines top-K experts for each input token
- Gating scores activate subset of experts per token

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities to prevent overloading specific experts

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Interleaving strategy**: While one batch processes on GPU, next batch transfers from other nodes
- **Implementation**: CUDA streams or asynchronous communication libraries (NCCL/MPI) ensure data transfer doesn't block GPU computation

### 3.2 Pipeline Scheduling
- **Multi-layer coordination**:
  - Token outputs from previous MoE layer immediately routed to next layer's experts
  - Experts in subsequent layers start processing partial batches without waiting for full batch
- **Fine-grained pipeline**: Increases throughput and reduces expert idle time

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Large Expert Parallelism = EP ≥ 16
- **Primary limiting factor**: Network bandwidth (mitigated by topology-aware routing and token batching)
- **One-expert-per-GPU policy**: Ensures all GPUs fully utilized for compute while communication costs amortized across many tokens

### 4.2 Memory and Model Parallelism Integration
- **Tensor Model Parallelism (TP)**: Applied within single expert's GPU if expert cannot fit on one GPU
- **Data Parallelism (DP)**: Applied across replicas of MoE network for synchronized weight updates while maintaining high expert-level parallelism
- **Integration**: Seamless combination with TP and DP for models exceeding single-GPU memory

## 5. Summary of Technical Advantages
1. **Maximized Expert Parallelism**: One expert per GPU ensures minimal contention and high compute efficiency
2. **Balanced Load Across Nodes**: Topology-aware placement and dynamic gating prevent network bottlenecks
3. **Scalable Communication Overlap**: Asynchronous token routing enables near-linear scaling for EP ≥ 16
4. **Compatibility with Large Models**: Integrates with TP and DP for models exceeding single-GPU memory

## 6. Implementation Details
- **Precision**: FP16
- **Communication libraries**: NCCL or MPI for asynchronous communication
- **Scheduling**: CUDA streams for overlapping computation and communication
- **Load monitoring**: Real-time per-expert load tracking for dynamic gating adjustment
- **Topology awareness**: Network bandwidth and latency metrics for expert placement decisions