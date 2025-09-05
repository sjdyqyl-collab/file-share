# Phase 2: Methodology Extraction

## Methods Overview

The proposed method maximizes expert-level parallelism in large-scale Mixture-of-Experts (MoE) models through three key components:

1. **Expert Placement Strategy** - Physical assignment of experts to GPUs
2. **Routing and Load Balancing** - Dynamic token distribution
3. **Communication Overlap and Scheduling** - Optimizing data transfer

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: At most one expert per GPU
- **Assignment rule**: 
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Experts replicated across GPUs while maintaining concurrency
- **Benefit**: Eliminates intra-GPU expert contention

### 1.2 Cross-Node Distribution Algorithm
- **Topology-aware placement** considers:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- **Standard MoE routing**: Top-K gating scores determine expert activation
- **K value**: Not explicitly stated (typically K=2 for MoE)

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches async to overlap with computation
- **Load Balancing**: Dynamic adjustment of gating probabilities to prevent expert overload

### 2.3 Load Balancing Algorithm
- **Monitoring**: Per-expert load tracking
- **Adjustment**: Dynamic gating probability modification
- **Prevention**: Avoid overloading specific experts

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Mechanism**: Interleave expert computation with token transfers
- **Implementation**: 
  - While batch N processes on GPU, batch N+1 transfers simultaneously
  - CUDA streams or NCCL/MPI for async communication
- **Benefit**: Data transfer does not block GPU computation

### 3.2 Pipeline Scheduling
- **Multi-layer coordination**:
  - Token outputs from layer L immediately route to layer L+1 experts
  - Subsequent layer experts start processing partial batches
- **Fine-grained pipeline**: Reduces expert idle time

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Large EP = configurations with 16+ experts per parallel group
- **Primary limiting factor**: Network bandwidth
- **Mitigation strategies**:
  - Topology-aware routing
  - Token batching optimization

### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Applied within expert if FFN exceeds GPU memory
- **Data Parallelism (DP)**: Synchronized weight updates across MoE replicas
- **Integration**: Maintains high expert-level parallelism while handling large models

## 5. Implementation Details

### 5.1 Hardware Requirements
- **GPU**: H100-class GPUs with high-bandwidth interconnects
- **Network**: NVLink, InfiniBand, or H100-class NVSwitch fabrics
- **Scale**: Minimum 16 GPUs for large EP regime

### 5.2 Software Stack
- **Communication**: NCCL or MPI for cross-node communication
- **Scheduling**: CUDA streams for async operations
- **Routing**: Custom gating network with load balancing

## 6. Method Summary

The methodology achieves:
1. **Maximal Expert Parallelism**: One expert per GPU ensures minimal contention
2. **Balanced Load**: Topology-aware placement prevents network bottlenecks
3. **Scalable Communication**: Async routing enables near-linear scaling for EP ≥ 16
4. **Large Model Support**: Integrates with TP and DP for memory-constrained scenarios

## 7. Critical Implementation Parameters

- **Expert Parallelism degree**: EP ≥ 16
- **Experts per layer**: 16 (in experiment)
- **Total experts**: 64 (4 layers × 16 experts)
- **GPU count**: 64 (one expert per GPU)
- **Token batch size**: 1024 tokens per forward pass
- **Precision**: FP16
- **Hidden dimension**: 32768 (MLP)
- **Attention heads**: 16 heads × 512 dimensions = 8192 total