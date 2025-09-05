# Phase 2: Methodology Extraction

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: At most one expert per GPU
- **Mathematical Constraint**: For E experts and G GPUs, assign each expert to distinct GPU if E ≤ G
- **Replication Strategy**: If E > G, replicate experts across GPUs to maximize independent expert concurrency while balancing memory usage
- **Benefit**: Eliminates intra-GPU contention, fully utilizes GPU compute units

### 1.2 Cross-Node Distribution Algorithm
- **Topology-Aware Placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU principle

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- **Standard MoE Architecture**: Top-K gating scores determine expert activation per token
- **K Value**: Not explicitly stated, but standard practice is K=2

### 2.2 Token Sharding Strategy
1. **Token Batching**: Group tokens by destination expert to reduce network messages
2. **Asynchronous Routing**: Send token batches asynchronously to overlap expert computation
3. **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

### 2.3 Communication Pattern
- **Input**: Tokens distributed based on gating decisions
- **Output**: Processed tokens returned to requesting GPUs
- **Optimization**: Token batches sized to maximize network efficiency

## 3. Communication Overlap and Scheduling

### 3.1 Compute-Communication Overlap
- **Mechanism**: Interleave expert computation with cross-node token transfers
- **Implementation**: 
  - CUDA streams for asynchronous operations
  - NCCL/MPI for cross-node communication
  - While batch N processes on GPU, batch N+1 transfers simultaneously

### 3.2 Pipeline Scheduling
- **Multi-layer MoE Networks**:
  - Token outputs from layer L immediately routed to layer L+1 experts
  - Experts start processing partial batches without waiting for full batch completion
  - Fine-grained pipeline increases throughput and reduces idle time

## 4. Memory and Model Parallelism Integration

### 4.1 Tensor Model Parallelism (TP)
- **Application**: Within expert if single expert's FFN cannot fit on one GPU
- **Configuration**: Optional TP=2 for very large experts
- **Scope**: Limited to within-expert parallelism only

### 4.2 Data Parallelism (DP)
- **Purpose**: Synchronized weight updates across MoE network replicas
- **Integration**: Maintains high expert-level parallelism while enabling training

### 4.3 Pipeline Parallelism (PP)
- **Structure**: Each MoE layer as a micro-stage
- **Communication**: Token transfer between layers overlapped with computation

## 5. Large EP Regime Optimization (EP ≥ 16)

### 5.1 Network Bandwidth Management
- **Primary Constraint**: Network bandwidth in large EP regime
- **Mitigation**: 
  - Topology-aware routing
  - Token batching optimization
  - Overlapping communication with computation

### 5.2 Resource Utilization
- **Compute**: All GPUs fully utilized for expert computation
- **Memory**: Balanced usage through expert distribution
- **Network**: Amortized communication costs across many tokens

## 6. Implementation Details

### 6.1 Hardware Requirements
- **GPUs**: H100-class with high-bandwidth interconnects
- **Network**: NVLink, InfiniBand, or NVSwitch fabric
- **Scale**: Designed for 64+ GPU clusters

### 6.2 Software Stack
- **Communication**: NCCL, MPI
- **Scheduling**: CUDA streams for asynchronous operations
- **Monitoring**: Per-expert load tracking for dynamic balancing