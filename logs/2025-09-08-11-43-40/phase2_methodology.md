# Phase 2: Methodology Extraction

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: Deploy at most one expert per GPU
- **Implementation**: 
  - For E experts and G GPUs: assign each expert to distinct GPU if E ≤ G
  - If E > G: replicate experts across GPUs maximizing concurrency while balancing memory
- **Benefit**: Each expert processes tokens without contention from other experts on same device

### 1.2 Cross-Node Distribution
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard MoE top-K gating scores determine expert activation per token
- Dynamic routing based on learned gating network

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
- **Load Balancing**: 
  - Monitor per-expert load
  - Dynamically adjust gating probabilities to prevent expert overloading
  - Ensure balanced workload distribution

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Interleaving Strategy**:
  - While current token batch processes on GPU, next batch transfers from other nodes
  - CUDA streams or asynchronous communication (NCCL/MPI) to prevent blocking
- **Implementation**: Non-blocking data transfer during GPU computation

### 3.2 Pipeline Scheduling
- **Multi-layer MoE coordination**:
  - Token outputs immediately routed to next layer's experts
  - Subsequent layer experts start processing partial batches without waiting for full batch
- **Fine-grained pipeline**: Increases throughput and reduces expert idle time

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Large Expert Parallelism = EP ≥ 16
- **Network bandwidth**: Primary limiting factor, mitigated by topology-aware routing and token batching
- **Compute utilization**: One-expert-per-GPU ensures full GPU utilization while amortizing communication costs

### 4.2 Memory and Model Parallelism Integration
- **Tensor Model Parallelism (TP)**: Applied within single expert if FFN cannot fit on one GPU (optional TP=2)
- **Data Parallelism (DP)**: Applied across MoE network replicas for synchronized weight updates
- **Compatibility**: Seamless integration with TP and DP for models exceeding single-GPU memory

## 5. System Architecture Summary

### 5.1 Parallelism Dimensions
- **Expert Parallelism (EP)**: Primary dimension, maximized to ≥16
- **Tensor Parallelism (TP)**: Optional within expert (TP=2 if needed)
- **Pipeline Parallelism (PP)**: Each MoE layer as micro-stage
- **Data Parallelism (DP)**: Across model replicas

### 5.2 Communication Patterns
- **Token routing**: Cross-node token transfers based on expert placement
- **Gradient synchronization**: DP all-reduce across replicas
- **Tensor parallelism**: All-reduce within expert if TP>1

### 5.3 Resource Utilization
- **GPU utilization**: 100% compute utilization per expert
- **Memory efficiency**: Balanced expert placement prevents memory hotspots
- **Network efficiency**: Topology-aware routing minimizes cross-node traffic