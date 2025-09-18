# Phase 2: Methodology Extraction

## Overview
Our approach focuses on maximizing expert-level parallelism in large-scale Mixture-of-Experts (MoE) models by deploying at most one expert per GPU, and distributing experts across nodes to exploit available compute resources fully. The core idea is to shift the computational bottleneck from intra-GPU contention to communication, which is effectively mitigated through careful scheduling, routing, and overlapping of communication and computation.

## Core Components

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: At most one expert per GPU
- **Mathematical Formulation**: For E experts and G GPUs, ensure each expert is assigned to a distinct GPU if E ≤ G
- **Replication Strategy**: If E > G, replicate experts across GPUs to maximize concurrency of independent experts while balancing memory usage
- **Benefit**: Each expert processes tokens without contention from other experts on the same device

#### 1.2 Cross-Node Distribution
- **Topology-Aware Placement** considers:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize the maximum number of tokens sent across any single link while maintaining one-expert-per-GPU principle

### 2. Routing and Load Balancing

#### 2.1 Gating Mechanism
- **Standard MoE Routing**: Top-K gating scores determine expert activation for each input token
- **Dynamic Adjustment**: Monitor per-expert load and adjust gating probabilities to prevent overloading

#### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network message count
- **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
- **Load Balancing**: Ensure balanced workload across all experts to prevent stragglers

### 3. Communication Overlap and Scheduling

#### 3.1 Overlapping Compute and Communication
- **Interleaving Strategy**: While one batch processes on GPU, next batch transfers from other nodes
- **Implementation**: CUDA streams or asynchronous communication libraries (NCCL/MPI)
- **Benefit**: Data transfer does not block GPU computation

#### 3.2 Pipeline Scheduling
- **Layer-to-Layer Routing**: Token outputs from previous MoE layer immediately routed to next layer's experts
- **Partial Batch Processing**: Experts start processing as soon as partial batch arrives, not waiting for full batch
- **Throughput Gain**: Fine-grained pipeline increases throughput and reduces idle time

### 4. Scalability Considerations

#### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Large Expert Parallelism when EP ≥ 16
- **Network Bottleneck**: Bandwidth becomes primary limiting factor
- **Mitigation**: Topology-aware routing and token batching
- **Compute Policy**: One-expert-per-GPU ensures full GPU utilization while amortizing communication costs

#### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Applied within single expert if FFN cannot fit on one GPU (optional TP=2)
- **Data Parallelism (DP)**: Applied across replicas of MoE network for synchronized weight updates
- **Compatibility**: Seamless integration with TP and DP for models exceeding single-GPU memory

## Technical Specifications

### Model Architecture Parameters
- **Layers**: 4-layer MoE
- **Experts per Layer**: 16
- **Expert Type**: MLP (Multi-Layer Perceptron)
- **Token Dimension**: 8192
- **MHA Configuration**: 16 heads × 512 dimensions = 8192 total
- **MLP Hidden Size**: 32768
- **Precision**: FP16

### Deployment Configurations

#### Baseline Configuration
- **Parallelism**: TP=8, PP=2
- **GPUs**: 16 H100
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (2 stages total)
  - Experts colocated: 4 experts per GPU

#### Proposed Configuration
- **Parallelism**: EP=64 (one expert per GPU)
- **GPUs**: 64 H100
- **Per-GPU Allocation**:
  - Each GPU hosts exactly one expert
  - Optional TP=2 within expert if needed
  - Each MoE layer as micro-stage with overlapped communication
- **Routing**: Dynamic token routing to GPU holding corresponding expert with asynchronous batching

## Implementation Details

### Communication Strategy
- **Token Transfer**: Asynchronous cross-node token routing
- **Batching**: Group tokens by destination expert
- **Overlap**: CUDA streams for concurrent computation and communication
- **Scheduling**: Fine-grained pipeline between MoE layers

### Load Balancing Algorithm
- **Monitoring**: Per-expert load tracking
- **Adjustment**: Dynamic gating probability modification
- **Balancing**: Prevent expert overloading and stragglers
- **Optimization**: Minimize network congestion through balanced distribution