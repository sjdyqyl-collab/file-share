# Phase 2: Methodology Extraction

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Core Principle**: Deploy at most one expert per GPU
- **Implementation**: For E experts and G GPUs, assign each expert to a distinct GPU when E ≤ G
- **When E > G**: Replicate experts across GPUs to maximize concurrency while balancing memory
- **Benefit**: Eliminates intra-GPU contention, fully utilizes GPU compute units

### 1.2 Cross-Node Distribution
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard MoE gating network determines top-K gating scores for each input token
- Top-K experts are activated based on these scores

### 2.2 Token Sharding Across Nodes
1. **Token Batching**: Group tokens by destination expert to reduce network messages
2. **Asynchronous Routing**: Send token batches asynchronously to overlap with computation
3. **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Mechanism**: Interleave expert computation and communication
- **Implementation**: 
  - Process one batch while transferring next batch simultaneously
  - Use CUDA streams or asynchronous communication (NCCL/MPI)
  - Ensure data transfer doesn't block GPU computation

### 3.2 Pipeline Scheduling
- **For multi-layer MoE networks**:
  - Token outputs from previous layer immediately routed to next layer's experts
  - Experts start processing partial batches as soon as they arrive
  - Avoid waiting for full batch completion

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Expert Parallelism degree ≥ 16
- **Characteristics**:
  - Network bandwidth becomes primary limiting factor
  - One-expert-per-GPU ensures all GPUs utilized for compute
  - Communication costs amortized across many tokens

### 4.2 Memory and Model Parallelism Integration
- **For large models exceeding single-GPU memory**:
  - Apply tensor model parallelism (TP) within each expert's GPU if needed
  - Apply data parallelism (DP) across MoE network replicas
  - Maintain synchronized weight updates while preserving expert-level parallelism

## 5. Implementation Details

### 5.1 Model Configuration
- **Layers**: 4 MoE layers
- **Experts per layer**: 16 (total 64 experts)
- **Expert type**: MLP
- **Precision**: FP16
- **Input**: 1024 sequences × 10000 tokens
- **Attention**: 16 heads × 512 dimensions = 8192 total dimensions
- **MLP hidden size**: 32768

### 5.2 Communication Strategy
- **Asynchronous token routing** using NCCL/MPI
- **Token batching** by destination expert
- **Topology-aware routing** based on cluster interconnect
- **Overlap computation** with communication using CUDA streams

### 5.3 Load Balancing Algorithm
- **Dynamic adjustment** of gating probabilities based on:
  - Current expert load
  - Historical routing patterns
  - Network congestion metrics
- **Monitoring**: Real-time per-expert load tracking
- **Adjustment**: Proportional gating probability modification