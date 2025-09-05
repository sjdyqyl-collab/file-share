# Large-Scale Cross-Node Expert Parallelism - Detailed Methodology

## Overview
The method consists of three key components: Expert Placement Strategy, Routing and Load Balancing, and Communication Overlap and Scheduling. The core principle is to shift the computational bottleneck from intra-GPU contention to communication overhead, which can be optimized through careful scheduling.

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
**Constraint**: Deploy at most one expert per GPU

**Mathematical Formulation**:
- Let E = number of experts in a MoE layer
- Let G = number of available GPUs
- If E ≤ G: Each expert assigned to distinct GPU
- If E > G: Experts replicated across GPUs with maximum concurrency

**Benefits**:
- Eliminates intra-GPU resource contention
- Maximizes GPU compute unit utilization
- Enables true expert-level parallelism

### 1.2 Cross-Node Distribution
**Topology-Aware Placement Algorithm**:

**Inputs**:
- Node-to-node bandwidth matrix B[i,j]
- Node-to-node latency matrix L[i,j]
- GPU memory capacity per node M[k]
- Expected token routing patterns P[e]

**Objective**:
Minimize: max_link_load = max(tokens_sent_across_link[i,j])
Subject to:
- One expert per GPU constraint
- Memory capacity constraints M[k]
- Expert replication constraints (when E > G)

**Placement Algorithm**:
```
1. Calculate expert communication patterns based on P[e]
2. Sort experts by communication volume
3. Place highest-volume experts on nodes with highest bandwidth
4. Balance remaining experts to minimize max_link_load
5. Ensure memory constraints are satisfied
```

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
**Standard MoE gating**:
- For each token t, compute gating scores g[t,e] for all experts e
- Select top-K experts (typically K=2)
- Route token to selected experts

### 2.2 Token Sharding Across Nodes
**Token Batching Process**:
1. **Grouping**: Group tokens by destination expert
   - Input: Token stream T = {t1, t2, ..., tn}
   - Output: Expert batches B[e] = {tokens destined for expert e}

2. **Asynchronous Routing**:
   - Send B[e] to GPU hosting expert e
   - Overlap with current expert computation
   - Use non-blocking communication primitives

3. **Load Balancing**:
   - Monitor expert load L[e] = tokens_processed[e]/time_window
   - Adjust gating probabilities: g'[t,e] = g[t,e] * α[e]
   - Where α[e] = min(1, target_load/L[e])

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
**Implementation Details**:
- **CUDA Streams**: Separate streams for computation and communication
- **NCCL Operations**: ncclSend, ncclRecv for peer-to-peer transfers
- **Double Buffering**: Prepare next batch while computing current batch

**Timeline**:
```
Time:    |----Compute Batch 1----|
Stream1: |----Expert Compute----|
Stream2:         |----Send Results----|----Recv Batch 2----|
```

### 3.2 Pipeline Scheduling
**Multi-layer MoE Scheduling**:
- Each MoE layer = micro-stage in pipeline
- Token routing: output of layer i → input of layer i+1
- Fine-grained scheduling: process partial batches as they arrive

**Pipeline Stages**:
1. **Receive**: Get tokens from previous layer/node
2. **Route**: Apply gating, batch tokens by expert
3. **Compute**: Expert computation on GPU
4. **Send**: Route results to next layer experts

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
**Characteristics**:
- Network bandwidth becomes primary limiting factor
- Communication cost amortized across many tokens
- One-expert-per-GPU ensures full GPU utilization

**Optimization Strategies**:
- Topology-aware routing
- Token batching to reduce message count
- Hierarchical communication patterns

### 4.2 Memory and Model Parallelism Integration
**Tensor Model Parallelism (TP) within Expert**:
- Applied when single expert FFN cannot fit on one GPU
- TP=2 typically sufficient for 32768 hidden dimension

**Data Parallelism (DP) across Replicas**:
- DP applied across complete MoE network replicas
- Synchronized weight updates
- Maintains high expert-level parallelism

## 5. Implementation Details

### 5.1 Communication Library Requirements
- **NCCL**: For NVIDIA GPU collectives
- **MPI**: For cross-node communication
- **CUDA Runtime**: For asynchronous operations

### 5.2 Memory Management
- **Expert Weights**: Stored in GPU memory (FP16: 2 bytes per parameter)
- **Activation Buffers**: Double buffering for overlap
- **Communication Buffers**: Pre-allocated pinned memory

### 5.3 Load Monitoring
- **Metrics Collected**:
  - Expert utilization per time window
  - Network link utilization
  - GPU memory usage
  - Token routing distribution

## 6. Algorithm Summary

**High-Level Algorithm**:
```
Initialize:
  - Place experts according to topology-aware algorithm
  - Setup communication patterns
  - Initialize load balancers

For each batch:
  1. Route tokens to experts using gating network
  2. Batch tokens by destination expert
  3. Asynchronously send token batches
  4. Overlap computation with communication
  5. Collect results and route to next layer
  6. Update load balancing parameters
```

## Critical Parameters for Deployment
- **EP degree**: Must be ≥ 16 for large EP regime
- **Experts per layer**: 64 in experimental setup
- **Hidden dimension**: 32768 for MLP experts
- **Precision**: FP16 throughout
- **Batch size**: 1024 sequences × 10000 tokens
- **Communication**: NCCL with CUDA streams
- **Placement**: Topology-aware with one expert per GPU