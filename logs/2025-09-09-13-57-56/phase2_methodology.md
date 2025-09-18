# Phase 2: Methodology Extraction

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: At most one expert per GPU
- **Assignment Rule**: 
  - If E ≤ G (experts ≤ GPUs): Each expert assigned to distinct GPU
  - If E > G: Experts replicated across GPUs to maximize concurrency
- **Benefit**: Each expert processes tokens without contention from other experts

### 1.2 Cross-Node Distribution
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard top-K gating scores determine expert activation per token
- K value not explicitly specified in paper

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with computation
- **Load Balancing**: Dynamic adjustment of gating probabilities to prevent expert overload

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Mechanism**: Interleave expert computation with token transfers
- **Implementation**: CUDA streams or asynchronous communication libraries (NCCL/MPI)
- **Process**: While batch N processes, batch N+1 transfers simultaneously

### 3.2 Pipeline Scheduling
- **Layer Coordination**: Token outputs immediately routed to next layer's experts
- **Micro-staging**: Each MoE layer acts as micro-stage
- **Partial Processing**: Experts start processing as soon as partial batch arrives

## 4. Scalability Framework

### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Expert Parallelism degree of 16 or more
- **Primary Limitation**: Network bandwidth (mitigated via topology-aware routing)
- **Compute Saturation**: One-expert-per-GPU ensures full GPU utilization

### 4.2 Integration with Other Parallelism
- **Tensor Parallelism (TP)**: Applied within expert if single expert exceeds GPU memory
- **Data Parallelism (DP)**: Applied across MoE network replicas
- **Pipeline Parallelism (PP)**: Each layer as micro-stage with overlapped communication

## 5. Memory and Compute Specifications

### 5.1 Model Architecture Details
- **Layers**: 4 MoE layers
- **Experts per Layer**: 16
- **Expert Type**: MLP
- **Token Dimension**: 8192
- **MLP Hidden Size**: 32768
- **MHA Configuration**: 16 heads × 512 dimensions per head = 8192 total

### 5.2 Precision and Batch Configuration
- **Precision**: FP16
- **Batch Size**: 1024 sequences
- **Sequence Length**: 10000 tokens per sequence
- **Total Tokens per Batch**: 10,240,000 tokens