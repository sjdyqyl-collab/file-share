# Phase 2: Methodology Extraction

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Methodology

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: Deploy at most one expert per GPU
- **Rule**: For E experts and G GPUs:
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Replicate experts across GPUs to maximize concurrency while balancing memory
- **Benefit**: Each expert processes tokens without contention from other experts on same device

#### 1.2 Cross-Node Distribution Algorithm
- **Inputs**: 
  - E experts per layer
  - G total GPUs available
  - Network topology (bandwidth, latency between nodes)
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU
- **Algorithm**: Topology-aware placement considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns

### 2. Routing and Load Balancing

#### 2.1 Gating Mechanism
- **Standard MoE routing**: Top-K gating scores determine which experts are activated for each token
- **K value**: Not explicitly stated in paper (typically K=2 for MoE models)

#### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities to avoid overloading specific experts

#### 2.3 Routing Process
1. For each input token, compute gating scores for all experts
2. Select top-K experts based on gating scores
3. Batch tokens by destination expert/node
4. Asynchronously send token batches to target GPUs
5. Receive processed tokens back from experts

### 3. Communication Overlap and Scheduling

#### 3.1 Overlapping Compute and Communication
- **Mechanism**: Interleave expert computation and communication
- **Implementation**: 
  - While one batch processes on GPU, next batch transfers from other nodes
  - CUDA streams or asynchronous communication libraries (NCCL/MPI) for non-blocking data transfer

#### 3.2 Pipeline Scheduling
- **Multi-layer MoE networks**:
  - Token outputs from previous MoE layer immediately routed to next layer's experts
  - Experts in subsequent layers start processing partial batches rather than waiting for full batch
- **Micro-staging**: Each MoE layer acts as a micro-stage in pipeline

### 4. Scalability Considerations

#### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Expert Parallelism degree ≥ 16
- **Network bottleneck**: Bandwidth becomes primary limiting factor
- **Mitigation**: Topology-aware routing and token batching
- **Compute utilization**: One-expert-per-GPU ensures all GPUs fully utilized

#### 4.2 Memory and Model Parallelism Integration
- **Tensor Model Parallelism (TP)**: Applied within expert if single expert cannot fit on one GPU
  - Optional TP=2 mentioned in experiments
- **Data Parallelism (DP)**: Applied across replicas of MoE network
  - Synchronized weight updates while maintaining high expert-level parallelism

### 5. Mathematical Formulation

#### 5.1 Expert Assignment
- Let E = number of experts per layer
- Let G = number of GPUs
- Expert assignment matrix A ∈ {0,1}^(E×G) where A[i,j] = 1 if expert i assigned to GPU j
- Constraint: Σ_i A[i,j] ≤ 1 for all j (at most one expert per GPU)

#### 5.2 Communication Cost
- Let T = total tokens per batch
- Let B = token batch size
- Communication rounds = ceil(T/B) × number of expert activations
- Objective: Minimize total communication time through overlapping and batching

### 6. Implementation Details

#### 6.1 Hardware Requirements
- **GPUs**: H100-class with high-bandwidth interconnects
- **Network**: NVLink, InfiniBand, or NVSwitch fabrics
- **Memory**: Sufficient GPU memory to hold one expert per GPU

#### 6.2 Software Stack
- **Communication**: NCCL or MPI for cross-node communication
- **Scheduling**: CUDA streams for asynchronous operations
- **Load Balancing**: Dynamic gating probability adjustment

#### 6.3 Memory Layout
- Each GPU stores:
  - Exactly one expert (MLP with hidden_size=32768)
  - Token buffers for incoming/outgoing tokens
  - Gating network parameters (shared across all GPUs)