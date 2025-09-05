# Phase 2: Methodology Extraction

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: Each GPU hosts at most one expert
- **Mathematical formulation**: For E experts and G GPUs, ensure each expert on distinct GPU if E ≤ G
- **Replication strategy**: When E > G, replicate experts to maximize concurrency while balancing memory
- **Resource isolation**: Each expert processes tokens without contention from other experts on same device

### 1.2 Cross-Node Distribution
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link
- **Constraint**: Maintain one-expert-per-GPU principle

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- **Top-K gating**: Standard MoE gating network determines expert activation
- **K value**: Not explicitly stated, but typically K=2 for MoE models
- **Dynamic adjustment**: Monitor per-expert load and adjust gating probabilities

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously while overlapping computation
- **Load Balancing**: Prevent overloading specific experts through dynamic adjustment

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Mechanism**: Interleave expert computation with token transfers
- **Implementation**: CUDA streams or asynchronous communication (NCCL/MPI)
- **Pipeline**: While batch N processed, batch N+1 transferred simultaneously

### 3.2 Pipeline Scheduling
- **Layer-wise processing**: Each MoE layer as micro-stage
- **Immediate routing**: Token outputs routed to next layer's experts immediately
- **Partial batch processing**: Start processing as soon as partial batch arrives

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Expert Parallelism degree ≥ 16
- **Primary limiting factor**: Network bandwidth
- **Mitigation**: Topology-aware routing and token batching
- **Compute utilization**: All GPUs fully utilized for compute

### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Applied within expert if FFN exceeds single GPU memory
- **Data Parallelism (DP)**: Across replicas of MoE network
- **Synchronization**: Synchronized weight updates while maintaining expert-level parallelism

## 5. Model Architecture Specifications

### 5.1 MoE Model Configuration
- **Layers**: 4 MoE layers
- **Experts per layer**: 16 experts
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16
- **Batch size**: 1024 tokens per forward pass

### 5.2 Transformer Specifications
- **MHA (Multi-Head Attention)**:
  - Number of heads: 16
  - Dimension per head: 512
- **MLP hidden size**: 32768
- **Total parameters**: Not explicitly stated, but can be calculated from dimensions

## 6. Implementation Details

### 6.1 Hardware Requirements
- **GPU type**: H100
- **GPU count**: 64 for proposed method, 16 for baseline
- **Network**: NVLink, InfiniBand, NVSwitch fabric
- **Memory**: Sufficient for one expert per GPU

### 6.2 Software Stack
- **Communication libraries**: NCCL, MPI
- **CUDA streams**: For asynchronous operations
- **Precision**: FP16 for computation efficiency