# Phase 2: Methodology Extraction

## Methods Overview

Our approach focuses on maximizing expert-level parallelism in large-scale Mixture-of-Experts (MoE) models by deploying at most one expert per GPU, and distributing experts across nodes to exploit available compute resources fully. The core idea is to shift the bottleneck from inter-expert contention to network communication, which can be mitigated through careful scheduling, routing, and overlapping of communication and computation.

## Core Methodology Components

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: Deploy at most one expert per GPU
- **Mathematical Constraint**: For E experts and G GPUs, ensure each expert assigned to distinct GPU when E ≤ G
- **Replication Strategy**: If E > G, replicate experts across GPUs to maximize concurrency while balancing memory
- **Compute Isolation**: Each expert processes tokens without contention from other experts on same device

#### 1.2 Cross-Node Distribution
- **Topology-Aware Placement**: Considers node-to-node bandwidth, latency, GPU memory capacity per node, and expected token routing patterns
- **Optimization Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU principle
- **Placement Algorithm**: Distributes experts across nodes to minimize hotspotting

### 2. Routing and Load Balancing

#### 2.1 Gating Mechanism
- **Standard MoE Routing**: Top-K gating scores determine expert activation per input token
- **Dynamic Adjustment**: Monitor per-expert load and adjust gating probabilities to prevent overloading

#### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap expert computation
- **Load Balancing**: Ensure balanced workload across all experts to prevent stragglers

### 3. Communication Overlap and Scheduling

#### 3.1 Overlapping Compute and Communication
- **Interleaving Strategy**: Process one token batch on GPU while transferring next batch from other nodes
- **Implementation**: CUDA streams or asynchronous communication libraries (NCCL/MPI) for non-blocking data transfer
- **Synchronization**: Ensure data transfer doesn't block GPU computation

#### 3.2 Pipeline Scheduling
- **Multi-layer Optimization**: Token outputs from previous MoE layer immediately routed to next layer's experts
- **Partial Batch Processing**: Experts start processing as soon as partial batch arrives rather than waiting for full batch
- **Fine-grained Pipeline**: Increases throughput and reduces expert idle time

### 4. Scalability Considerations

#### 4.1 Large EP Regime (EP ≥ 16)
- **Network Bandwidth Focus**: Primary limiting factor in large EP setups
- **Topology-Aware Routing**: Mitigate bandwidth limitations through careful routing and token batching
- **One-Expert-Per-GPU**: Ensures all GPUs fully utilized for compute while communication costs amortized

#### 4.2 Memory and Model Parallelism Integration
- **Tensor Model Parallelism (TP)**: Partition individual experts using TP within GPU if expert exceeds memory
- **Data Parallelism (DP)**: Apply DP across MoE network replicas for synchronized weight updates
- **Compatibility**: Seamless integration with TP and DP for models exceeding single-GPU memory

## Technical Implementation Details

### GPU Allocation Strategy
- **Maximum Expert Distribution**: Ensure EP ≥ 16 for large EP qualification
- **Memory Utilization**: Balance expert placement with GPU memory constraints
- **Compute Saturation**: Maximize GPU utilization through expert isolation

### Communication Patterns
- **Cross-Node Token Transfer**: Efficient routing of tokens to distributed experts
- **Asynchronous Communication**: Overlap computation with token transfers
- **Bandwidth Optimization**: Minimize network congestion through token batching

### Load Balancing Mechanisms
- **Dynamic Gating Adjustment**: Real-time adjustment of expert selection probabilities
- **Token Distribution Monitoring**: Track per-expert load for balanced processing
- **Straggler Prevention**: Prevent individual expert overload through proactive load balancing