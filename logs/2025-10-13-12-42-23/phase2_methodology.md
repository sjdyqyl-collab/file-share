# Phase 2: Methodology Extraction

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Core Principle**: Deploy at most one expert per GPU to maximize expert-level parallelism
- **Mathematical Constraint**: For E experts and G GPUs, ensure each expert is assigned to a distinct GPU if E ≤ G
- **Expert Replication**: If E > G, replicate experts across GPUs to maximize concurrency while balancing memory usage
- **Resource Utilization**: Each expert processes tokens without contention from other experts on the same device

### 1.2 Cross-Node Distribution
- **Topology-Aware Placement**: Consider node-to-node bandwidth, latency, GPU memory capacity, and expected token routing patterns
- **Optimization Objective**: Minimize maximum number of tokens sent across any single link while maintaining one-expert-per-GPU principle
- **Hotspot Prevention**: Distribute experts across nodes to minimize computational hotspots

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- **Top-K Selection**: Routing governed by top-K gating scores for each input token
- **Expert Activation**: Subset of experts activated based on gating network output

### 2.2 Token Sharding Across Nodes
1. **Token Batching**: Group tokens by destination expert to reduce network message count
2. **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
3. **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

### 2.3 Load Balancing Algorithm
- **Monitoring**: Continuous tracking of per-expert computational load
- **Adjustment**: Dynamic modification of gating probabilities to ensure balanced workload distribution
- **Straggler Prevention**: Prevent specific experts from becoming performance bottlenecks

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Interleaved Execution**: Process current batch while transferring next batch simultaneously
- **Technology Stack**: CUDA streams, NCCL, or MPI for asynchronous communication
- **Non-blocking Transfer**: Ensure data transfer does not block GPU computation

### 3.2 Pipeline Scheduling for Multi-layer MoE
- **Immediate Routing**: Token outputs from previous MoE layer immediately routed to next layer's experts
- **Partial Batch Processing**: Experts start processing as soon as partial batch arrives, not waiting for full batch
- **Fine-grained Pipeline**: Increases throughput and reduces expert idle time

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Network Bandwidth Focus**: Primary limiting factor becomes network bandwidth
- **Mitigation Strategies**: Topology-aware routing and token batching
- **Compute Utilization**: One-expert-per-GPU ensures full GPU utilization while amortizing communication costs

### 4.2 Memory and Model Parallelism Integration
- **Tensor Model Parallelism (TP)**: Partition individual experts within GPU when model exceeds single-GPU memory
- **Data Parallelism (DP)**: Apply across MoE network replicas for synchronized weight updates
- **Hybrid Approach**: Maintain high expert-level parallelism while handling very large models

## 5. Implementation Details

### 5.1 Hardware Requirements
- **GPU Specification**: H100-class GPUs with high-bandwidth interconnects
- **Network Infrastructure**: NVLink, InfiniBand, or H100-class NVSwitch fabrics
- **Cluster Environment**: High-performance computing (HPC) environments

### 5.2 Software Stack
- **Communication Libraries**: NCCL, MPI for cross-node communication
- **CUDA Streams**: For asynchronous compute-communication overlap
- **Routing Middleware**: Custom implementation for token routing and load balancing

## 6. Mathematical Formulations

### 6.1 Expert Placement Optimization
```
Minimize: max_link_traffic
Subject to:
  - one_expert_per_GPU_constraint
  - memory_capacity_constraint
  - bandwidth_availability_constraint
```

### 6.2 Load Balancing Objective
```
Balance: expert_load_distribution
Target: minimize_max_expert_load
Method: dynamic_gating_adjustment
```

## 7. Key Advantages Summary
1. **Maximized Expert Parallelism**: One expert per GPU ensures minimal contention
2. **Balanced Load Distribution**: Topology-aware placement prevents bottlenecks
3. **Scalable Communication**: Asynchronous routing enables near-linear scaling
4. **Large Model Compatibility**: Seamless integration with TP and DP strategies