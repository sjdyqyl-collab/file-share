# Methodology: Large-Scale Cross-Node Expert Parallelism

## 1. Overview
The method maximizes expert-level parallelism by distributing experts across GPUs with at most one expert per GPU. This shifts the optimization focus from communication reduction to compute concurrency maximization.

## 2. Expert Placement Strategy

### 2.1 Single-Expert-Per-GPU Deployment
- **Principle**: Each GPU hosts at most one expert
- **Mathematical formulation**: For E experts and G GPUs:
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Experts replicated across GPUs to maximize concurrency
- **Benefit**: Eliminates intra-GPU contention between experts

### 2.2 Cross-Node Distribution
- **Topology-aware placement** considers:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

## 3. Routing and Load Balancing

### 3.1 Gating Mechanism
- Standard top-K gating scores determine expert activation per token
- K value typically 1 or 2 for sparse activation

### 3.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously while overlapping expert computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities

## 4. Communication Overlap and Scheduling

### 4.1 Overlapping Compute and Communication
- **Interleaving strategy**: Process current batch while transferring next batch
- **Implementation**: CUDA streams or NCCL/MPI for asynchronous communication
- **Non-blocking**: Data transfer doesn't block GPU computation

### 4.2 Pipeline Scheduling
- **Layer-wise pipeline**: Each MoE layer treated as micro-stage
- **Immediate routing**: Token outputs routed to next layer's experts immediately
- **Partial batch processing**: Experts start processing as soon as partial batch arrives

## 5. Scalability Considerations

### 5.1 Large EP Regime (EP ≥ 16)
- **Definition**: Expert Parallelism degree ≥ 16 qualifies as "large EP"
- **Network bottleneck**: Bandwidth becomes primary limiting factor
- **Mitigation**: Topology-aware routing and token batching
- **Utilization**: One-expert-per-GPU ensures full GPU utilization

### 5.2 Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Applied within expert if FFN exceeds single-GPU memory
  - Optional TP=2 for very large experts
- **Data Parallelism (DP)**: Applied across MoE network replicas
- **Synchronization**: Synchronized weight updates while maintaining expert-level parallelism

## 6. Implementation Details

### 6.1 Hardware Requirements
- **GPUs**: H100-class with high-bandwidth interconnects
- **Network**: NVLink, InfiniBand, or H100 NVSwitch fabrics
- **Scale**: Designed for clusters with ≥16 GPUs

### 6.2 Software Stack
- **Communication**: NCCL or MPI for cross-node communication
- **Scheduling**: CUDA streams for async operations
- **Monitoring**: Per-expert load tracking for dynamic balancing

## 7. Mathematical Formulation

### 7.1 Expert Placement Optimization
```
minimize: max_link_load = max_{i,j} (tokens_sent_{i→j})
subject to: 
  - 1 expert per GPU constraint
  - GPU memory capacity constraints
  - Network topology constraints
```

### 7.2 Load Balancing Objective
```
minimize: load_variance = Σ_k (tokens_to_expert_k - mean_load)²
subject to:
  - Gating probability constraints
  - Expert capacity constraints
```

## 8. Summary of Methodological Advantages
1. **Maximized Expert Parallelism**: One expert per GPU ensures minimal contention
2. **Balanced Load**: Topology-aware placement prevents network bottlenecks
3. **Scalable Communication**: Async routing enables near-linear scaling for EP ≥ 16
4. **Model Compatibility**: Seamless integration with TP and DP for large models