# Phase 2: Methodology Extraction

## Method Overview
Three key components for maximizing expert-level parallelism:
1. Expert Placement Strategy
2. Routing and Load Balancing
3. Communication Overlap and Scheduling

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: At most one expert per GPU
- **Implementation**: 
  - For E experts and G GPUs: assign each expert to distinct GPU if E ≤ G
  - If E > G: replicate experts across GPUs while maximizing independent expert concurrency
- **Benefit**: Each expert processes tokens without contention from other experts

### 1.2 Cross-Node Distribution Algorithm
- **Topology-aware placement** considers:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- **Standard MoE routing**: Top-K gating scores determine expert activation per token
- **K value**: Not explicitly stated (typically K=2 in MoE literature)

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Mechanism**: Interleave expert computation and communication
- **Implementation**: 
  - While current batch processes on GPU, next batch transfers from other nodes
  - Use CUDA streams or asynchronous communication (NCCL/MPI)
  - Ensure data transfer doesn't block GPU computation

### 3.2 Pipeline Scheduling
- **Multi-layer MoE networks**:
  - Token outputs from previous layer immediately routed to next layer's experts
  - Experts in subsequent layers start processing partial batches without waiting for full batch
- **Benefit**: Fine-grained pipeline increases throughput and reduces idle time

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Expert Parallelism degree ≥ 16
- **Optimization focus**: 
  - Network bandwidth becomes primary limiting factor
  - Mitigated through topology-aware routing and token batching
  - One-expert-per-GPU ensures full GPU utilization while amortizing communication costs

### 4.2 Memory and Model Parallelism Integration
- **Tensor Model Parallelism (TP)**: Applied within single expert if FFN exceeds GPU memory
- **Data Parallelism (DP)**: Applied across MoE network replicas for synchronized weight updates
- **Constraint**: TP only used when necessary (optional TP=2 mentioned in experiments)

## 5. Implementation Details

### 5.1 Hardware Requirements
- **GPUs**: H100-class with high-bandwidth interconnects
- **Network**: NVLink, InfiniBand, or H100-class NVSwitch fabrics
- **Memory**: Sufficient per-GPU memory for single expert

### 5.2 Software Stack
- **Communication Libraries**: NCCL or MPI for asynchronous communication
- **Scheduling**: CUDA streams for overlapping compute and communication
- **Load Balancing**: Dynamic monitoring and adjustment of gating probabilities

## 6. Mathematical Formulation

### 6.1 Expert Placement
- **Variables**: 
  - E = number of experts per layer
  - G = number of GPUs
  - N = number of nodes
- **Constraint**: Expert placement matrix P ∈ {0,1}^(G×E) where P[i,j]=1 if expert j on GPU i
- **Objective**: Minimize max link utilization while ensuring Σ_j P[i,j] ≤ 1 for all i

### 6.2 Load Balancing
- **Token distribution**: For token t, routing probability p_t(e) for expert e
- **Load balancing**: Adjust p_t(e) based on current expert utilization u_e
- **Constraint**: Maintain top-K routing while balancing load