# Phase 2: Methodology Extraction

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: At most one expert per GPU
- **Assignment Rule**: 
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Replicate experts across GPUs to maximize concurrency while balancing memory
- **Benefit**: No intra-GPU expert contention, full utilization of GPU compute units

### 1.2 Cross-Node Distribution
- **Topology-Aware Placement** considers:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU principle

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard MoE top-K gating network
- For each token, top-K gating scores determine activated experts

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously while overlapping expert computation
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Interleaving Strategy**: While one batch processes on GPU, next batch transfers from other nodes
- **Implementation**: CUDA streams or asynchronous communication libraries (NCCL/MPI)
- **Benefit**: Data transfer doesn't block GPU computation

### 3.2 Pipeline Scheduling
- **Multi-layer MoE Coordination**:
  - Token outputs from previous MoE layer immediately routed to next layer's experts
  - Experts in subsequent layers start processing on partial batch arrival
- **Advantage**: Fine-grained pipeline increases throughput, reduces idle time

## 4. Scalability Framework

### 4.1 Large EP Regime (EP ≥ 16)
- **Network Bandwidth**: Primary limiting factor
- **Mitigation**: Topology-aware routing and token batching
- **One-expert-per-GPU**: Ensures full GPU utilization while amortizing communication across many tokens

### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Applied within GPU if single expert FFN cannot fit
- **Data Parallelism (DP)**: Applied across MoE network replicas for synchronized weight updates
- **Integration**: Maintains high expert-level parallelism while handling large models

## 5. Implementation Details

### 5.1 Hardware Configuration
- **GPUs**: H100 clusters
- **Network**: NVLink, InfiniBand, H100-class NVSwitch fabrics
- **Advantage**: Advanced interconnects make communication cost less dominant

### 5.2 Software Stack
- **Communication Libraries**: NCCL, MPI for asynchronous operations
- **Scheduling**: CUDA streams for compute-communication overlap
- **Monitoring**: Per-expert load tracking for dynamic balancing

## 6. Methodology Summary
- **Core Principle**: Shift optimization from reducing communication to maximizing compute concurrency
- **Key Enabler**: Modern HPC networking capabilities sustain high bandwidth and low latency
- **Design Focus**: Large EP setups with careful cluster topology alignment
- **Performance Goal**: Near-linear scaling in large MoE deployments