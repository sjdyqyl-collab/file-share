# Phase 2: Methodology Extraction - Large-Scale Cross-Node Expert Parallelism

## Method Overview
The method maximizes expert-level parallelism in MoE models by deploying at most one expert per GPU, distributing experts across nodes to fully exploit available compute resources. This shifts the bottleneck from intra-GPU contention to communication, which is mitigated through careful scheduling, routing, and overlapping of communication and computation.

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: Each GPU hosts at most one expert
- **Implementation**:
  - For E experts and G GPUs: assign each expert to distinct GPU if E ≤ G
  - If E > G: replicate experts across GPUs to maximize concurrency while balancing memory
  - Ensures each expert processes tokens without contention from other experts on same device

### 1.2 Cross-Node Distribution
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard MoE gating network determines expert activation
- Top-K gating scores determine which experts are activated per input token

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities to prevent overloading specific experts

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Mechanism**: Interleave expert computation and communication
- **Implementation**:
  - While one batch processes on GPU, next batch transfers from other nodes simultaneously
  - Use CUDA streams or asynchronous communication libraries (NCCL/MPI)
  - Ensure data transfer doesn't block GPU computation

### 3.2 Pipeline Scheduling
- **Multi-layer MoE networks**:
  - Token outputs from previous MoE layer immediately routed to next layer's experts
  - Experts in subsequent layers start processing partial batches without waiting for full batch
  - Fine-grained pipeline increases throughput and reduces idle time

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Network bandwidth** becomes primary limiting factor
- **Mitigation**: Topology-aware routing and token batching
- **One-expert-per-GPU** ensures all GPUs fully utilized for compute
- **Communication costs** amortized across many tokens

### 4.2 Memory and Model Parallelism Integration
- **Tensor Model Parallelism (TP)**: Applied within GPU if single expert's FFN cannot fit
- **Data Parallelism (DP)**: Applied across MoE network replicas for synchronized weight updates
- **Expert-level parallelism** maintained while handling large models

## 5. Implementation Parameters

### Model Configuration
- **Layers**: 4 MoE layers
- **Experts per layer**: 16
- **Expert type**: MLP
- **Hidden size**: 32768
- **Precision**: FP16
- **Batch size**: 1024 tokens per forward pass
- **MHA**: 16 heads × 512 dimensions per head

### Deployment Requirements
- **Minimum GPUs**: 64 H100 for full deployment
- **Interconnect**: NVLink, InfiniBand, or H100-class NVSwitch
- **Communication**: NCCL or MPI for asynchronous transfers
- **Scheduling**: CUDA streams for computation-communication overlap