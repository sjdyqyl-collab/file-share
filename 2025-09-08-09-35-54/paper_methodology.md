# Large-Scale Cross-Node Expert Parallelism - Methodology

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Policy**: At most one expert per GPU
- **Assignment Logic**:
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Experts replicated across GPUs to maximize independent expert concurrency
- **Memory Optimization**: Each expert runs in isolation without contention

### 1.2 Cross-Node Distribution Algorithm
- **Inputs**: 
  - E = number of experts per layer (16)
  - N = number of nodes
  - G = GPUs per node
  - Network topology (bandwidth/latency matrix)
- **Objective**: Minimize max tokens sent across any single link
- **Constraints**: One expert per GPU, balanced memory usage

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- **Top-K Selection**: K=2 experts per token (standard MoE)
- **Gating Function**: Softmax over expert scores
- **Dynamic Adjustment**: Monitor per-expert load and adjust gating probabilities

### 2.2 Token Sharding Across Nodes
- **Token Batching**:
  - Group tokens by destination expert
  - Batch size = 1024 sequences × variable tokens per sequence
  - Reduce network messages through aggregation
- **Asynchronous Routing**:
  - Send token batches while experts compute current batch
  - Use CUDA streams for non-blocking transfers
- **Load Balancing Metrics**:
  - Track tokens per expert per batch
  - Adjust gating weights to prevent expert overload

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Dual Buffering**:
  - Buffer A: Current token batch being processed
  - Buffer B: Next token batch being transferred
- **CUDA Stream Management**:
  - Stream 0: Expert computation
  1. Stream 1: Token transfers (NCCL)
  - Synchronization: Event-based triggers

### 3.2 Pipeline Scheduling for Multi-Layer MoE
- **Layer-wise Pipeline**:
  - Each MoE layer = pipeline stage
  - Token outputs immediately routed to next layer
- **Micro-batching**:
  - Split 1024 sequences into micro-batches
  - Process partial batches to reduce idle time
- **Dependencies**:
  - Layer N+1 starts processing when first tokens arrive from Layer N

## 4. Scalability Framework

### 4.1 Large EP Regime (EP ≥ 16)
- **Network Requirements**:
  - InfiniBand/NVLink for high bandwidth
  - Minimize all-to-all communication patterns
- **Load Distribution**:
  - Uniform expert placement across nodes
  - Avoid hotspotting on specific nodes

### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism within Expert**:
  - Optional TP=2 if expert FFN > GPU memory
  - Apply only to individual expert, not across experts
- **Data Parallelism**:
  - DP replicas across MoE networks
  - Synchronized weight updates
  - Expert placement consistent across replicas

## 5. Implementation Details

### 5.1 Expert Architecture
- **Structure**: MLP with GELU activation
- **Dimensions**:
  - Input: 8192 (token dimension)
  - Hidden: 32768
  - Output: 8192
- **Precision**: FP16 throughout

### 5.2 Communication Patterns
- **All-to-All**: Token routing between experts
- **All-Reduce**: Gradient synchronization in DP
- **Point-to-Point**: Expert-to-expert token transfers

### 5.3 Scheduling Algorithm
```
for each layer in [1, 2, 3, 4]:
    for each expert in [1..16]:
        if tokens_arrived(expert):
            compute_expert(expert, tokens)
        if computation_complete(expert):
            route_tokens_to_next_layer(expert)
```

## 6. Baseline Comparison Setup
- **Baseline Configuration**:
  - TP=8, PP=2
  - 16 H100 GPUs
  - 4 experts per GPU
  - Sequential pipeline processing
- **Proposed Configuration**:
  - EP=64 (16 experts × 4 layers)
  - 64 H100 GPUs
  - 1 expert per GPU
  - Overlapped communication and computation