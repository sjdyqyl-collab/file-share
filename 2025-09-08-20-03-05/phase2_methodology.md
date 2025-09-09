# Phase 2: Methodology Extraction

## Methods Overview

The proposed method focuses on maximizing expert-level parallelism in large-scale Mixture-of-Experts (MoE) models by deploying at most one expert per GPU, and distributing experts across nodes to exploit available compute resources fully. The core idea is to shift the computational bottleneck from intra-GPU contention to network communication, which is effectively mitigated through careful scheduling, routing, and overlapping of communication and computation.

## Detailed Methodology

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: Deploy at most one expert per GPU to eliminate intra-GPU contention
- **Mathematical formulation**: For a MoE layer with E experts and a cluster of G GPUs, ensure each expert is assigned to a distinct GPU if E ≤ G
- **Replication strategy**: If E > G, replicate experts across GPUs to maximize concurrency of independent experts while balancing memory usage
- **Benefit**: Each expert processes tokens without contention from other experts on the same device, fully utilizing GPU compute units

#### 1.2 Cross-Node Distribution
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency characteristics
  - GPU memory capacity per node
  - Expected token routing patterns from gating mechanism
- **Objective**: Minimize the maximum number of tokens sent across any single link while maintaining the one-expert-per-GPU principle

### 2. Routing and Load Balancing

#### 2.1 Gating Mechanism
- **Standard MoE routing**: Top-K gating scores determine which subset of experts is activated for each token
- **K value**: Not explicitly specified, but typically K=1 or K=2 in MoE architectures

#### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce the number of network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities to avoid overloading specific experts

### 3. Communication Overlap and Scheduling

#### 3.1 Overlapping Compute and Communication
- **Mechanism**: While one batch of tokens is being processed on a GPU, the next batch is simultaneously transferred from other nodes
- **Implementation**: CUDA streams or asynchronous communication libraries (NCCL or MPI) ensure data transfer does not block GPU computation

#### 3.2 Pipeline Scheduling
- **Multi-layer coordination**: Token outputs from previous MoE layer are immediately routed to next layer's experts
- **Fine-grained processing**: Experts in subsequent layers start processing as soon as a partial batch arrives, rather than waiting for the full batch
- **Throughput improvement**: Increases throughput and reduces idle time for each expert

### 4. Scalability Considerations

#### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Large Expert Parallelism where EP ≥ 16
- **Network bottleneck**: Network bandwidth becomes primary limiting factor
- **Mitigation**: Topology-aware routing and token batching
- **One-expert-per-GPU**: Ensures all GPUs are fully utilized for compute while communication costs are amortized across many tokens

#### 4.2 Memory and Model Parallelism Integration
- **Tensor Model Parallelism (TP)**: Applied within individual experts if they cannot fit on a single GPU
- **Data Parallelism (DP)**: Applied across replicas of the MoE network for synchronized weight updates
- **Compatibility**: Seamlessly integrates with TP and DP for models exceeding single-GPU memory

### 5. Implementation Details

#### 5.1 Critical Parameters
- **Expert count**: 64 experts per layer (16 experts × 4 layers)
- **GPU allocation**: 64 H100 GPUs (1 expert per GPU)
- **Precision**: FP16
- **Token dimension**: 8192
- **Hidden size**: 32768 (MLP hidden dimension)
- **Sequence length**: 10000 tokens
- **Batch size**: 1024 sequences

#### 5.2 Communication Patterns
- **Token routing**: Dynamic routing based on gating scores
- **Cross-node transfers**: Asynchronous token batch transfers between nodes
- **Overlap strategy**: Computation-communication overlap using CUDA streams

### 6. Summary of Advantages
1. **Maximized Expert Parallelism**: One expert per GPU ensures minimal contention and high compute efficiency
2. **Balanced Load Across Nodes**: Topology-aware placement and dynamic gating prevent network bottlenecks
3. **Scalable Communication Overlap**: Asynchronous token routing allows near-linear scaling for EP ≥ 16
4. **Compatibility with Large Models**: Integrates seamlessly with TP and DP for models exceeding single-GPU memory