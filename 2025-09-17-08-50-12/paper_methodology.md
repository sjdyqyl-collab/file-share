# Methodology Extraction - Large-Scale Cross-Node Expert Parallelism

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: At most one expert per GPU
- **Assignment Rule**: 
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Experts replicated across GPUs while maximizing independent expert concurrency
- **Objective**: Ensure each expert processes tokens without contention from other experts on same device

### 1.2 Cross-Node Distribution Algorithm
- **Inputs**: 
  - E: Number of experts per layer
  - G: Total GPUs available
  - Network topology (bandwidth, latency matrix)
  - GPU memory capacity per node
  - Expected token routing patterns
- **Placement Objective**: Minimize max(tokens sent across any single link) while maintaining one-expert-per-GPU
- **Topology-Aware Strategy**: 
  - Consider node-to-node bandwidth and latency
  - Balance GPU memory usage across nodes
  - Account for expected token routing distributions

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- **Standard MoE gating**: Top-K gating scores determine expert activation
- **K value**: Not explicitly stated (typically K=1 or K=2 in MoE models)
- **Dynamic adjustment**: Monitor per-expert load and adjust gating probabilities

### 2.2 Token Sharding Across Nodes
- **Token Batching**: 
  - Group tokens by destination expert
  - Reduce number of network messages
  - Batch size optimization for network efficiency
- **Asynchronous Routing**: 
  - Send token batches asynchronously
  - Overlap with expert computation
  - Minimize idle time
- **Load Balancing Algorithm**:
  1. Monitor per-expert load continuously
  2. Detect overloaded experts via queue length/completion time
  3. Adjust gating probabilities to redirect tokens
  4. Rebalance without affecting model accuracy

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Mechanism**: Interleave expert computation and communication
- **Implementation**:
  - While batch i processed on GPU j
  - Batch i+1 transferred to GPU j simultaneously
- **Technology Stack**:
  - CUDA streams for GPU-level parallelism
  - NCCL/MPI for cross-node communication
  - Asynchronous communication primitives

### 3.2 Pipeline Scheduling for Multi-Layer MoE
- **Token Flow**: 
  - Immediate routing from layer l to layer l+1
  - No waiting for full batch completion
- **Fine-grained Pipeline**:
  - Partial batch processing starts as soon as tokens arrive
  - Reduce idle time per expert
  - Overlap across multiple MoE layers

## 4. Scalability Framework

### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Expert Parallelism degree ≥ 16
- **Characteristics**:
  - Network bandwidth becomes primary limiting factor
  - One-expert-per-GPU ensures full GPU utilization
  - Communication costs amortized across many tokens
- **Optimization Focus**: Topology-aware routing and token batching

### 4.2 Memory and Model Parallelism Integration
- **Tensor Model Parallelism (TP)**:
  - Applied within single expert if expert exceeds GPU memory
  - Intra-expert partitioning using standard TP techniques
- **Data Parallelism (DP)**:
  - Applied across replicas of entire MoE network
  - Synchronized weight updates
  - Maintains high expert-level parallelism
- **Integration Hierarchy**:
  1. Expert Parallelism (EP) - across experts
  2. Tensor Parallelism (TP) - within experts (if needed)
  3. Data Parallelism (DP) - across model replicas
  4. Pipeline Parallelism (PP) - across layers (if needed)

## 5. Implementation Details

### 5.1 Hardware Requirements
- **GPU**: H100-class GPUs recommended
- **Network**: High-bandwidth interconnects (NVLink, InfiniBand, NVSwitch)
- **Scale**: Designed for 16+ GPUs (EP ≥ 16)

### 5.2 Software Stack
- **Communication**: NCCL, MPI
- **GPU Programming**: CUDA streams
- **Framework**: Compatible with existing MoE implementations
- **Scheduling**: Custom load balancer and router

### 5.3 Memory Management
- **Per-GPU Memory**: Single expert per GPU reduces memory contention
- **Token Buffering**: Asynchronous token transfer requires buffer space
- **Gradient Storage**: DP requires gradient synchronization across replicas

## 6. Algorithmic Flow

### 6.1 Forward Pass Algorithm
```
For each MoE layer:
  1. Route tokens using gating network
  2. Batch tokens by destination expert
  3. Initiate asynchronous token transfers
  4. While waiting for tokens:
     - Process locally available tokens
     - Prepare next batch for transfer
  5. Process received tokens on destination experts
  6. Aggregate results and route to next layer
```

### 6.2 Load Balancing Algorithm
```
Continuous monitoring:
  1. Track per-expert queue lengths
  2. Measure expert processing times
  3. Identify overloaded/underloaded experts
  4. Adjust gating probabilities:
     - Reduce probability for overloaded experts
     - Increase probability for underloaded experts
  5. Ensure adjustment maintains model accuracy
```

## 7. Optimization Parameters

### 7.1 Communication Parameters
- **Token batch size**: Optimize for network packet efficiency
- **Transfer granularity**: Balance latency vs. throughput
- **Buffer sizes**: Minimize memory overhead while preventing stalls

### 7.2 Load Balancing Parameters
- **Monitoring frequency**: Balance overhead vs. responsiveness
- **Adjustment rate**: Smooth load transitions
- **Threshold values**: Define overloaded/underloaded conditions

### 7.3 Scheduling Parameters
- **Pipeline depth**: Number of concurrent operations
- **Overlap ratio**: Compute vs. communication time balance
- **Synchronization points**: Minimize global barriers