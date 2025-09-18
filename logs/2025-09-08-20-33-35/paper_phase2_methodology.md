# Phase 2: Methodology Extraction

## 1. Overview
The method focuses on maximizing expert-level parallelism in large-scale MoE models by deploying at most one expert per GPU, distributing experts across nodes to exploit available compute resources fully. The approach shifts the bottleneck from inter-expert contention to network communication, mitigated through careful scheduling, routing, and overlapping of computation and communication.

## 2. Expert Placement Strategy

### 2.1 Single-Expert-Per-GPU Deployment
- **Principle**: Deploy at most one expert per GPU
- **Implementation**: 
  - For MoE layer with E experts and cluster of G GPUs: assign each expert to distinct GPU if E ≤ G
  - If E > G: replicate experts across GPUs to maximize concurrency of independent experts while balancing memory usage
- **Benefit**: Each expert processes tokens without contention from other experts on same device, fully utilizing GPU compute units

### 2.2 Cross-Node Distribution
- **Topology-aware placement strategy** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum number of tokens sent across any single link while maintaining one-expert-per-GPU principle

## 3. Routing and Load Balancing

### 3.1 Gating Mechanism
- Standard MoE routing: top-K gating scores determine which experts are activated for each input token

### 3.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap expert computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities to avoid overloading specific experts

## 4. Communication Overlap and Scheduling

### 4.1 Overlapping Compute and Communication
- **Interleaving Strategy**:
  - While one batch processes on GPU, next batch transfers simultaneously from other nodes
  - CUDA streams or asynchronous communication libraries (NCCL/MPI) ensure data transfer doesn't block GPU computation

### 4.2 Pipeline Scheduling for Multi-layer MoE
- **Fine-grained pipeline**:
  - Token outputs from previous MoE layer immediately routed to next layer's experts
  - Experts in subsequent layers start processing as soon as partial batch arrives (no waiting for full batch)
- **Benefit**: Increases throughput, reduces idle time for each expert

## 5. Scalability Considerations

### 5.1 Large EP Regime (EP ≥ 16)
- **Definition**: Large EP = configurations with 16 or more experts per parallel group
- **Characteristics**:
  - Network bandwidth becomes primary limiting factor
  - Mitigated by topology-aware routing and token batching
  - One-expert-per-GPU policy ensures all GPUs fully utilized for compute
  - Communication costs amortized across many tokens

### 5.2 Memory and Model Parallelism Integration
- **Tensor Model Parallelism (TP)**: Applied within GPU if single expert's FFN cannot fit (optional TP=2)
- **Data Parallelism (DP)**: Applied across replicas of MoE network for synchronized weight updates while maintaining high expert-level parallelism

## 6. Technical Specifications

### 6.1 Model Configuration
- **Layers**: 4-layer MoE
- **Experts per layer**: 16 (baseline) or 64 (proposed)
- **Expert type**: MLP
- **Precision**: FP16
- **Sequence parameters**:
  - Batch size: 1024 sequences
  - Sequence length: 10000 tokens
  - Token dimension: 8192
- **Attention parameters**:
  - MHA heads: 16
  - Head dimension: 512
- **MLP parameters**:
  - Hidden size: 32768

### 6.2 Deployment Configurations
- **Baseline**:
  - GPUs: 16 H100
  - TP=8, PP=2
  - 4 experts per GPU + TP shard
- **Proposed**:
  - GPUs: 64 H100 (one GPU per expert per layer)
  - 1 expert per GPU
  - Optional TP=2 for memory overflow
  - Pipeline: each MoE layer as micro-stage with overlapped communication