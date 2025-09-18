# Methodology: Large-Scale Cross-Node Expert Parallelism for MoE Models

## Overview
The methodology focuses on maximizing expert-level parallelism in large-scale MoE models by deploying at most one expert per GPU, distributing experts across nodes to fully exploit available compute resources. The approach shifts the bottleneck from inter-expert contention to network communication, which is mitigated through careful scheduling, routing, and overlapping of communication and computation.

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Policy**: Deploy at most one expert per GPU
- **Mathematical Constraint**: For MoE layer with E experts and cluster of G GPUs, ensure each expert assigned to distinct GPU if E ≤ G
- **Replication Strategy**: If E > G, replicate experts across GPUs to maximize concurrency while balancing memory usage
- **Benefit**: Each expert processes tokens without contention from other experts on same device

### 1.2 Cross-Node Distribution
- **Strategy**: Topology-aware placement considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum number of tokens sent across any single link while maintaining one-expert-per-GPU principle

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- **Standard MoE Approach**: Top-K gating scores determine which subset of experts activated for each token
- **Routing Decision**: Based on gating network output

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities to prevent overloading specific experts

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Interleaving Strategy**: Process one batch of tokens on GPU while simultaneously transferring next batch from other nodes
- **Implementation**: CUDA streams or asynchronous communication libraries (NCCL/MPI) ensure data transfer doesn't block GPU computation

### 3.2 Pipeline Scheduling
- **Multi-layer Processing**: Token outputs from previous MoE layer immediately routed to next layer's experts
- **Fine-grained Pipeline**: Experts in subsequent layers start processing as soon as partial batch arrives, rather than waiting for full batch
- **Throughput Increase**: Reduces idle time for each expert

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Network Bandwidth**: Primary limiting factor in large EP setups
- **Mitigation**: Topology-aware routing and token batching
- **One-expert-per-GPU Policy**: Ensures all GPUs fully utilized for compute while communication costs amortized across many tokens

### 4.2 Memory and Model Parallelism Integration
- **Tensor Model Parallelism (TP)**: Applied within single expert's FFN if cannot fit on one GPU (optional TP=2)
- **Data Parallelism (DP)**: Applied across replicas of MoE network for synchronized weight updates while maintaining high expert-level parallelism

## 5. Implementation Details

### 5.1 Model Configuration
- **Layers**: 4-layer MoE
- **Experts per Layer**: 16 (total 64 experts)
- **Expert Type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16
- **Token Dimension**: 8192
- **MLP Hidden Size**: 32768
- **MHA Configuration**: 16 heads, 512 dimension per head

### 5.2 Batch Configuration
- **Batch Size**: 1024 sequences
- **Sequence Length**: 10000 tokens per sequence
- **Total Tokens**: 10,240,000 tokens per batch

## 6. Deployment Architecture

### 6.1 Baseline Configuration (TP=8, PP=2)
- **GPUs**: 16 H100
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 tensor-parallel shard for all layers
  - Each pipeline stage (2 stages total) spans 8 GPUs
  - Experts colocated: typically 4 experts per GPU
- **Processing**: Tokens flow sequentially through pipeline stages, multiple experts per GPU share compute resources

### 6.2 Proposed Configuration
- **GPUs**: 64 H100 (one GPU per expert per layer)
- **Per-GPU Allocation**:
  - Each GPU hosts exactly one expert
  - Tensor parallelism applied only if single expert's FFN cannot fit on one GPU (optional TP=2)
  - Pipeline parallelism: each MoE layer is a micro-stage; communication of tokens overlapped with computation
- **Routing**: Input tokens dynamically routed to GPU holding corresponding expert, token batches asynchronously sent ensuring minimal idle time

## 7. Summary of Advantages
1. **Maximized Expert Parallelism**: One expert per GPU ensures minimal contention and high compute efficiency
2. **Balanced Load Across Nodes**: Topology-aware placement and dynamic gating prevent network bottlenecks
3. **Scalable Communication Overlap**: Asynchronous token routing allows near-linear scaling for EP ≥ 16
4. **Compatibility with Large Models**: Integrates seamlessly with TP and DP for models exceeding single-GPU memory