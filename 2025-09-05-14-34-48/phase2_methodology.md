# Phase 2: Methodology Extraction

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: Each GPU hosts at most one expert
- **Allocation Rule**: 
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Experts replicated across GPUs to maximize independent computation while balancing memory
- **Benefit**: Eliminates intra-GPU contention, full utilization of GPU compute units

### 1.2 Cross-Node Distribution Algorithm
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- **Standard MoE gating**: Top-K gating scores determine expert activation per token
- **Dynamic adjustment**: Monitor per-expert load and adjust gating probabilities to prevent overloading

### 2.2 Token Sharding Strategy
1. **Token Batching**: Group tokens by destination expert to reduce network messages
2. **Asynchronous Routing**: Send token batches asynchronously while overlapping expert computation
3. **Load Balancing**: Continuous monitoring and dynamic adjustment of expert loads

### 2.3 Cross-Node Token Transfer
- **Input tokens** dynamically routed to GPU holding corresponding expert
- **Token batches** asynchronously sent to minimize idle time
- **Communication pattern**: All-to-all communication for token redistribution

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Mechanism**: Interleave expert computation with cross-node token transfers
- **Implementation**: 
  - CUDA streams for asynchronous operations
  - NCCL/MPI for efficient cross-node communication
- **Pattern**: While GPU processes current batch, next batch is simultaneously transferred

### 3.2 Pipeline Scheduling for Multi-Layer MoE
- **Micro-staging**: Each MoE layer treated as micro-stage
- **Fine-grained pipeline**:
  - Token outputs from previous layer immediately routed to next layer's experts
  - Experts start processing partial batches as soon as they arrive
  - Eliminates waiting for full batch completion

### 3.3 Communication Patterns
- **All-to-all communication** for token redistribution between layers
- **Point-to-point communication** for expert-specific token routing
- **Topology-aware routing** to minimize network congestion

## 4. Scalability Framework

### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Expert Parallelism degree ≥ 16
- **Characteristics**:
  - Network bandwidth becomes primary limiting factor
  - One-expert-per-GPU ensures full GPU utilization
  - Communication costs amortized across many tokens

### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Applied within expert if single expert exceeds GPU memory
  - Optional TP=2 for memory-constrained scenarios
- **Data Parallelism (DP)**: Applied across MoE network replicas
- **Pipeline Parallelism (PP)**: Each MoE layer as micro-stage with overlapped communication

### 4.3 Resource Allocation Matrix
```
Layer 1: Expert 0-15 → GPU 0-15 (Node 0-3)
Layer 2: Expert 16-31 → GPU 16-31 (Node 4-7)
Layer 3: Expert 32-47 → GPU 32-47 (Node 8-11)
Layer 4: Expert 48-63 → GPU 48-63 (Node 12-15)
```

## 5. Implementation Details

### 5.1 Hardware Requirements
- **GPU**: H100-class GPUs with NVLink/InfiniBand interconnects
- **Network**: High-bandwidth, low-latency interconnects (NVSwitch fabrics)
- **Memory**: Sufficient GPU memory for single expert + activation buffers

### 5.2 Software Stack
- **Communication**: NCCL for GPU-to-GPU communication
- **Scheduling**: CUDA streams for asynchronous operations
- **Orchestration**: MPI for cross-node coordination
- **Memory management**: Unified memory for efficient token transfers

### 5.3 Load Balancing Algorithm
```
while training/inference:
    1. Monitor per-expert load distribution
    2. Calculate load imbalance metrics
    3. Adjust gating probabilities to balance load
    4. Redistribute tokens if necessary
    5. Overlap communication with computation
```

## 6. Integration Points

### 6.1 With Existing Parallelism Strategies
- **TP + EP**: Tensor parallelism within expert, expert parallelism across GPUs
- **PP + EP**: Pipeline stages per layer, experts distributed within stages
- **DP + EP**: Data parallel replicas with expert parallel distribution

### 6.2 Memory Optimization
- **Activation checkpointing**: Reduce memory footprint for long sequences
- **Gradient accumulation**: Handle large batch sizes across multiple iterations
- **Expert sharding**: Dynamic expert placement based on memory availability