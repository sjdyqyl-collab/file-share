# Large-Scale Cross-Node Expert Parallelism - Detailed Methodology

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Methodology

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: Each GPU hosts at most one expert
- **Mathematical Formulation**: For E experts and G GPUs, ensure each expert is assigned to a distinct GPU when E ≤ G
- **Replication Strategy**: When E > G, replicate experts across GPUs to maximize concurrency while balancing memory usage
- **Memory Allocation**: Each expert gets full GPU memory budget without sharing

#### 1.2 Cross-Node Distribution Algorithm
- **Input Parameters**:
  - Cluster topology: node-to-node bandwidth matrix B[i,j]
  - GPU memory capacity per node: M[node_id]
  - Expected token routing patterns: P[token → expert]
- **Objective**: Minimize max(tokens sent across any single link)
- **Constraints**: 
  - One-expert-per-GPU principle
  - GPU memory capacity limits
- **Output**: Expert-to-GPU mapping function f: expert_id → (node_id, gpu_id)

### 2. Routing and Load Balancing

#### 2.1 Gating Mechanism
- **Top-K Selection**: For each token, select top-K experts based on gating scores
- **Score Calculation**: softmax(W_gate * token_embedding)
- **K Value**: Typically K=2 for MoE models

#### 2.2 Token Sharding Algorithm
- **Token Batching**: Group tokens by destination expert
- **Batch Size Calculation**: B = ceil(total_tokens / num_experts)
- **Asynchronous Routing**: 
  - Send token batches asynchronously using CUDA streams
  - Overlap communication with computation
- **Load Balancing**: 
  - Monitor per-expert load L[expert_id]
  - Adjust gating probabilities: P'[expert] = P[expert] * (1 - α * (L[expert] - L_avg)/L_avg)

### 3. Communication Overlap and Scheduling

#### 3.1 Compute-Communication Overlap
- **Pipeline Stages**:
  1. Token routing and batching
  2. Asynchronous token transfer
  3. Expert computation
  4. Result aggregation
- **Overlap Strategy**: 
  - While GPU i processes batch j, simultaneously transfer batch j+1 to GPU i
  - Use separate CUDA streams for computation and communication

#### 3.2 Pipeline Scheduling for Multi-Layer MoE
- **Micro-batch Processing**: Split tokens into micro-batches of size m
- **Layer-wise Scheduling**: 
  - Layer l starts processing as soon as it receives m/2 tokens from layer l-1
  - No need to wait for full batch completion
- **Communication Pattern**: 
  - Point-to-point transfers between GPUs
  - NCCL or MPI for cross-node communication
  - Topology-aware routing to minimize hop count

### 4. Scalability Framework

#### 4.1 Large EP Regime (EP ≥ 16)
- **Network Requirements**: 
  - Minimum bandwidth: 50 GB/s per GPU (InfiniBand or NVLink)
  - Latency: < 5μs for same-node, < 15μs cross-node
- **Compute Saturation**: Ensure GPU utilization > 90%
- **Communication Overhead**: Amortized across many tokens

#### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Applied within expert if expert FFN > GPU memory
  - TP degree: 2 (split hidden dimension)
  - Communication: All-reduce after each linear layer
- **Data Parallelism (DP)**: Applied across replicas
  - DP degree: R (number of replicas)
  - Synchronized weight updates using all-reduce
- **Combined Parallelism**: DP × EP × TP × PP

### 5. Implementation Details

#### 5.1 Expert Computation Kernel
- **Input**: Token embeddings (batch_size, seq_len, hidden_dim)
- **Expert FFN**: Linear → GELU → Linear
- **Dimensions**:
  - Input: (1024, 10000, 8192)
  - Expert hidden: 32768
  - Output: (1024, 10000, 8192)

#### 5.2 Communication Protocol
- **Token Transfer Format**: 
  - Token embeddings: FP16 (batch_size, hidden_dim)
  - Routing indices: INT32 (batch_size, top_k)
- **Message Size**: batch_size × (8192 × 2 bytes + 4 bytes × K)
- **Transfer Protocol**: NCCL Send/Recv with CUDA IPC for same-node

#### 5.3 Load Balancing Monitor
- **Metrics Collected**:
  - Per-expert token count per batch
  - GPU utilization per expert
  - Network link utilization
- **Adjustment Frequency**: Every 100 batches
- **Convergence**: Target imbalance < 5% across experts