# Phase 2: Methodology Extraction

## Methods Overview
Our approach maximizes expert-level parallelism in MoE models by deploying at most one expert per GPU, distributing experts across nodes to fully exploit compute resources. The method shifts the bottleneck from inter-expert contention to network communication, mitigated through careful scheduling, routing, and overlapping communication/computation.

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Policy**: Deploy at most one expert per GPU
- **Condition**: For E experts and G GPUs, assign each expert to distinct GPU if E ≤ G
- **When E > G**: Replicate experts across GPUs to maximize concurrency while balancing memory
- **Benefit**: Each expert processes tokens without contention from other experts on same device

### 1.2 Cross-Node Distribution
- **Strategy**: Topology-aware placement considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard MoE gating network determines top-K experts per token

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities to prevent overloading

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Mechanism**: Interleave expert computation and communication
  - While batch N processes on GPU, batch N+1 transfers from other nodes
  - Use CUDA streams or asynchronous libraries (NCCL/MPI)
- **Implementation**: Non-blocking data transfer during GPU computation

### 3.2 Pipeline Scheduling
- **Multi-layer MoE**: 
  - Token outputs immediately routed to next layer's experts
  - Subsequent layer experts start processing partial batches without waiting for full batch completion
- **Benefit**: Fine-grained pipeline increases throughput, reduces idle time

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Expert Parallelism degree ≥ 16
- **Characteristics**:
  - Network bandwidth becomes primary limiting factor
  - Mitigated through topology-aware routing and token batching
  - One-expert-per-GPU ensures full GPU utilization while amortizing communication across tokens

### 4.2 Memory and Model Parallelism Integration
- **Large Models**: When single expert exceeds GPU memory
  - Apply tensor model parallelism (TP) within GPU if necessary
  - Use data parallelism (DP) across MoE network replicas
- **Synchronization**: Maintain synchronized weight updates while preserving high expert-level parallelism

## 5. Technical Specifications
- **Parallelism Types**: Combines EP, TP, PP, and DP
- **Communication Libraries**: NCCL, MPI
- **Hardware Requirements**: HPC networking (NVLink, InfiniBand, NVSwitch)
- **Memory Management**: Optional TP=2 for experts exceeding single-GPU memory
- **Scheduling**: Asynchronous token routing with CUDA stream management