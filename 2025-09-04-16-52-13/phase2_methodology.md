# Phase 2: Methodology Extraction

## Abstract (Retained from Original)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Methodology Overview

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Deployment
- **Core Principle**: Deploy at most one expert per GPU to eliminate intra-GPU contention
- **Mathematical Formulation**: For E experts and G GPUs, ensure each expert gets distinct GPU if E ≤ G
- **Replication Strategy**: When E > G, replicate experts across GPUs while maintaining memory balance
- **Memory Optimization**: Each expert has dedicated GPU memory, eliminating resource sharing overhead

#### 1.2 Cross-Node Distribution Algorithm
- **Topology Awareness**: Considers node-to-node bandwidth, latency, GPU memory capacity
- **Load Distribution**: Minimizes maximum tokens sent across any single network link
- **Placement Constraints**: 
  - One expert per GPU constraint
  - Network topology optimization
  - Expected token routing pattern analysis

### 2. Routing and Load Balancing System

#### 2.1 Gating Mechanism
- **Standard MoE Routing**: Top-K gating scores determine expert activation per token
- **K Value**: Typically K=2 for top-2 gating in standard MoE implementations
- **Dynamic Adjustment**: Gating probabilities adjusted based on per-expert load monitoring

#### 2.2 Token Sharding Strategy
- **Token Batching**: Groups tokens by destination expert to minimize network messages
- **Batch Size Optimization**: 1024 tokens per forward pass, batched by expert destination
- **Asynchronous Routing**: Non-blocking token transfers initiated before computation completion
- **Load Balancing Metrics**: 
  - Per-expert token count monitoring
  - Dynamic probability adjustment to prevent overloading
  - Straggler prevention through balanced distribution

### 3. Communication Overlap and Scheduling

#### 3.1 Compute-Communication Overlap
- **CUDA Streams**: Multiple streams for concurrent computation and communication
- **NCCL/MPI Integration**: Asynchronous communication primitives for cross-node transfers
- **Overlap Strategy**: 
  - While GPU processes current batch, next batch transfers in parallel
  - Zero-copy transfers where possible to minimize latency

#### 3.2 Pipeline Scheduling for Multi-Layer MoE
- **Micro-Stage Definition**: Each MoE layer treated as independent micro-stage
- **Token Routing Pipeline**: 
  - Immediate routing from layer n to layer n+1
  - Partial batch processing starts as soon as tokens arrive
  - Fine-grained pipelining reduces idle time
- **Synchronization Points**: 
  - Layer-wise synchronization barriers
  - Token completion tracking across distributed experts

### 4. Integration with Other Parallelism Strategies

#### 4.1 Tensor Model Parallelism (TP) Integration
- **Conditional Application**: TP applied only when single expert exceeds GPU memory
- **TP Configuration**: Optional TP=2 for experts with large parameter counts
- **Memory Calculation**: Expert parameter size = 32768 (hidden) × appropriate dimensions

#### 4.2 Data Parallelism (DP) Integration
- **Synchronized Updates**: DP applied across MoE network replicas
- **Gradient Synchronization**: All-reduce operations across DP groups
- **Expert Consistency**: Ensures expert weights remain synchronized across replicas

### 5. Scalability Framework

#### 5.1 Large EP Regime Optimization (EP ≥ 16)
- **Network Bandwidth Focus**: Primary optimization target in large EP regime
- **Communication Cost Amortization**: Spread across many tokens to reduce per-token overhead
- **Linear Scaling Target**: Achieve near-linear throughput scaling with GPU count

#### 5.2 Memory Management
- **Per-Expert Memory**: Dedicated GPU memory per expert eliminates sharing overhead
- **Dynamic Memory Allocation**: Adjust based on expert parameter requirements
- **Memory Footprint**: 32768 hidden size × parameter dimensions × FP16 precision

### 6. Implementation Details

#### 6.1 Hardware Requirements
- **GPU Specification**: H100-class GPUs with NVLink/NVSwitch interconnects
- **Network Infrastructure**: InfiniBand or equivalent high-bandwidth, low-latency connections
- **Memory Requirements**: Sufficient per-GPU memory for single expert + communication buffers

#### 6.2 Software Stack
- **Communication Libraries**: NCCL for GPU-to-GPU, MPI for cross-node coordination
- **Scheduling Framework**: Custom scheduler for token routing and expert assignment
- **Monitoring**: Real-time load balancing and performance metrics collection