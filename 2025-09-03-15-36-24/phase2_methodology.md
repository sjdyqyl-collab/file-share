# Phase 2: Methodology Extraction

## System Architecture Overview

### Parallelism Hierarchy
The proposed method integrates multiple parallelism dimensions:
- **Expert Parallelism (EP)**: 16-64 experts distributed across GPUs
- **Tensor Parallelism (TP)**: Applied within experts when memory constrained (TP=2 optional)
- **Pipeline Parallelism (PP)**: Each MoE layer as a micro-stage
- **Data Parallelism (DP)**: Across model replicas for training

### Core Principle
Shift from communication optimization to compute concurrency maximization by ensuring one expert per GPU.

## Detailed Methodology

### 1. Expert Placement Algorithm

#### 1.1 Single-Expert-Per-GPU Constraint
```
For E experts and G GPUs:
- If E ≤ G: Each expert assigned to distinct GPU
- If E > G: Replicate experts with load balancing
- Never colocate multiple experts on same GPU
```

#### 1.2 Topology-Aware Distribution
- Input: Cluster topology graph with bandwidth/latency metrics
- Output: Expert-to-GPU mapping minimizing maximum link utilization
- Constraints: 
  - GPU memory capacity ≥ expert size
  - One expert per GPU
  - Balanced token routing across nodes

### 2. Token Routing Mechanism

#### 2.1 Gating Network
- Top-K gating (K typically 2) per token
- Softmax over expert scores
- Load balancing loss: L_aux = α * Σ(f_i * P_i)
  - f_i: fraction of tokens routed to expert i
  - P_i: average probability of routing to expert i

#### 2.2 Token Sharding Process
1. **Batching Phase**: Group tokens by destination expert
2. **Routing Phase**: Asynchronous send using NCCL
3. **Computation Phase**: Expert processes received tokens
4. **Return Phase**: Send results back to source GPUs

### 3. Communication Optimization

#### 3.1 Overlap Strategy
- **Double Buffering**: Two token buffers per GPU
- **CUDA Streams**: Separate streams for compute and communication
- **Pipeline Depth**: 4 stages (send → compute → return → next layer)

#### 3.2 Scheduling Algorithm
```
For each MoE layer:
1. Prefetch tokens for next expert while current expert computes
2. Use all-reduce for gradient synchronization (if training)
3. Overlap token routing with attention computation in previous layer
```

### 4. Memory Management

#### 4.1 Expert Memory Layout
- Expert parameters: 32768 × hidden_dim × 2 bytes (FP16)
- Activation memory: batch_size × seq_len × hidden_dim
- Communication buffer: batch_size × seq_len × top_k × num_experts

#### 4.2 Memory Optimization
- Gradient checkpointing for training
- ZeRO optimizer states across DP replicas
- Expert parameter sharding within TP group

### 5. Load Balancing Dynamics

#### 5.1 Dynamic Adjustment
- Monitor queue lengths per expert every 100ms
- Adjust gating probabilities: p_i = softmax(logits_i + λ * load_i)
- λ = 0.1 for gradual adjustment

#### 5.2 Straggler Mitigation
- Timeout-based re-routing: 5ms threshold
- Backup expert activation for slow nodes
- Adaptive batch sizing based on expert load

### 6. Integration with Model Parallelism

#### 6.1 Tensor Parallelism Within Expert
- Applied only when expert FFN > GPU memory
- TP=2 splits hidden dimension 16384×2
- All-reduce within TP group after MLP

#### 6.2 Pipeline Parallelism
- Each MoE layer = 1 pipeline stage
- Micro-batch size = 256 tokens
- 4 micro-batches in flight simultaneously

## Implementation Details

### Hardware Requirements
- GPU: H100 with 80GB HBM3
- Network: InfiniBand HDR (200 Gbps) or NVSwitch
- CPU: 32 cores per node for orchestration

### Software Stack
- Framework: PyTorch with custom NCCL backend
- Communication: NCCL 2.18+ with SHARP support
- Scheduling: CUDA Graphs for deterministic execution

### Critical Parameters
- EP degree: 16-64 (must be ≥16 for large EP)
- Batch size: 1024 tokens
- Sequence length: 2048 (configurable)
- Top-K routing: K=2
- Expert capacity factor: 1.2
- Communication dtype: FP16 (same as compute)