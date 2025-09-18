# Phase 2: Methodology Extraction

## Methods Overview
Our approach focuses on maximizing expert-level parallelism in large-scale Mixture-of-Experts (MoE) models by deploying at most one expert per GPU, and distributing experts across nodes to exploit available compute resources fully.

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Configuration**: Each GPU hosts at most one expert
- **Constraint**: For E experts and G GPUs, ensure each expert assigned to distinct GPU if E ≤ G
- **Replication**: If E > G, replicate experts to maximize concurrency while balancing memory
- **Benefit**: Eliminates intra-GPU expert contention

### 1.2 Cross-Node Distribution
- **Topology-Aware Placement**: Considers node-to-node bandwidth, latency, GPU memory capacity, and token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- **Top-K Selection**: Standard MoE gating determines expert activation per token
- **Dynamic Adjustment**: Monitor per-expert load and adjust gating probabilities

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with computation
- **Load Balancing**: Prevent expert overloading through dynamic monitoring

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Interleaving Strategy**: Process one token batch while transferring the next
- **Implementation**: CUDA streams or asynchronous communication (NCCL/MPI)
- **Non-blocking**: Ensure data transfer doesn't block GPU computation

### 3.2 Pipeline Scheduling
- **Micro-staging**: Each MoE layer as a micro-stage
- **Immediate Routing**: Token outputs routed immediately to next layer's experts
- **Partial Processing**: Start processing as soon as partial batch arrives

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Network Bandwidth**: Primary limiting factor, mitigated by topology-aware routing
- **Compute Saturation**: One-expert-per-GPU ensures full GPU utilization
- **Communication Amortization**: Costs spread across many tokens

### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism**: Applied within single expert if needed (optional TP=2)
- **Data Parallelism**: Applied across MoE network replicas
- **Compatibility**: Seamless integration with TP and DP for large models

## 5. Model Architecture Specifications
- **Layers**: 4 MoE layers
- **Experts per Layer**: 16 experts
- **Expert Type**: MLP (Multi-Layer Perceptron)
- **Token Dimension**: 8192
- **Hidden Size of MLP**: 32768
- **MHA Configuration**: 16 heads, 512 dimensions per head
- **Precision**: FP16
- **Batch Configuration**: 1024 sequences × 10000 tokens per sequence