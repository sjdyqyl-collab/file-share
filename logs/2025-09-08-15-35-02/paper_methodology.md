# Detailed Methodology: Large-Scale Cross-Node Expert Parallelism

## Methodology Overview

The methodology focuses on maximizing expert-level parallelism by deploying at most one expert per GPU, distributing experts across nodes to fully exploit available compute resources. This shifts the optimization focus from reducing communication to maximizing compute concurrency.

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment

**Core Principle**: Each GPU hosts at most one expert to eliminate intra-GPU contention.

**Mathematical Formulation**:
- Let E = number of experts in MoE layer
- Let G = number of available GPUs
- Deployment rule: Each expert assigned to distinct GPU if E ≤ G
- If E > G: Replicate experts across GPUs to maximize concurrency while balancing memory usage

**Memory Allocation**:
- Each expert is an MLP with hidden size 32768
- Token dimension: 8192
- Expert parameters: 8192 × 32768 + 32768 × 8192 = 536,870,912 parameters per expert
- FP16 precision: 1,073,741,824 bytes (1GB) per expert

### 1.2 Cross-Node Distribution Algorithm

**Topology-Aware Placement**:
1. **Bandwidth Consideration**: Minimize tokens sent across low-bandwidth links
2. **Latency Optimization**: Place frequently co-activated experts on same node
3. **Memory Balancing**: Ensure equal distribution across all nodes
4. **Routing Pattern**: Account for expected token routing based on gating probabilities

**Placement Algorithm**:
```
For each MoE layer with E experts:
    1. Calculate optimal node distribution based on cluster topology
    2. Assign experts to GPUs ensuring one-expert-per-GPU
    3. If E > total GPUs, replicate experts with load balancing
    4. Verify memory constraints per GPU
```

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism Specifications

**Top-K Routing**:
- K = 2 (select top 2 experts per token)
- Gating network: Linear layer with 8192 input, 16 output (for 16 experts)
- Softmax activation for expert probabilities
- Load balancing loss: aux_loss = α * Σ(f_i * P_i) where f_i is fraction of tokens routed to expert i, P_i is average probability

### 2.2 Token Sharding Implementation

**Token Batching Process**:
1. **Input Processing**: 1024 sequences × 10000 tokens = 10,240,000 tokens per batch
2. **Gating Decision**: Each token routed to top-2 experts
3. **Batch Formation**: Group tokens by destination expert
4. **Size Calculation**: Each expert receives ~1,280,000 tokens (10,240,000 × 2 / 16)

**Asynchronous Routing Pipeline**:
```
Stage 1: Compute gating scores for all tokens
Stage 2: Sort tokens by destination expert
Stage 3: Package tokens into batches per expert
Stage 4: Asynchronously send token batches
Stage 5: Receive tokens and begin expert computation
```

### 2.3 Load Balancing Algorithm

**Dynamic Adjustment**:
- Monitor per-expert load every 1000 tokens
- Adjust gating probabilities: P'_i = P_i * (1 - λ * (load_i - avg_load))
- λ = 0.1 (balancing strength parameter)
- Constraint: ΣP'_i = 1.0 maintained

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication

**CUDA Stream Architecture**:
- Stream 0: Main computation stream
- Stream 1: Token sending operations
- Stream 2: Token receiving operations
- Stream 3: Gradient synchronization (if training)

**Overlap Schedule**:
```
Time t0: Start sending tokens for batch n+1
Time t1: Begin computation for batch n (overlaps with sending)
Time t2: Finish computation batch n, start receiving results
Time t3: Start computation for batch n+1 (results ready)
```

### 3.2 Pipeline Scheduling for Multi-Layer MoE

**Layer-by-Layer Processing**:
- 4 MoE layers total
- Each layer has 64 experts (16 per layer × 4 layers)
- Token routing between layers without global synchronization

**Fine-Grained Pipeline**:
```
Layer 1: Expert computation starts as soon as first tokens arrive
Layer 2: Begins processing when Layer 1 outputs first tokens
Layer 3: Begins when Layer 2 outputs available
Layer 4: Final processing with immediate output
```

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)

**Network Requirements**:
- Minimum bandwidth: 50 GB/s per GPU for token transfer
- Latency: <10 μs for optimal performance
- Topology: Fat-tree or dragonfly topology preferred

**Scaling Law**:
```
Throughput ∝ min(GPUs, Experts) when EP ≥ 16
Latency ∝ 1/EP + communication_overhead
```

### 4.2 Memory and Model Parallelism Integration

**Tensor Parallelism for Large Experts**:
- Applied only when single expert exceeds GPU memory
- TP=2 splits expert across 2 GPUs
- Communication overhead: 2× all-reduce operations

**Data Parallelism**:
- Replicate entire MoE model across multiple nodes
- Synchronize gradients using all-reduce
- Scale-out factor: Number of replicas = Total_GPUs / 64

## 5. Implementation Details

### 5.1 Expert Architecture

**MLP Structure**:
- Input projection: 8192 → 32768 (Linear)
- Activation: GELU
- Output projection: 32768 → 8192 (Linear)
- Dropout: 0.1 (training only)

### 5.2 Communication Protocol

**NCCL Configuration**:
- Algorithm: Ring all-reduce for gradient sync
- Protocol: Simple for small messages, LL for large messages
- Buffer size: 256MB per communication stream

**MPI Tags**:
- Token routing: Tag 100-115 (16 experts)
- Gradient sync: Tag 200-215
- Control messages: Tag 300-399

### 5.3 Memory Layout

**Per-GPU Memory Allocation**:
- Expert parameters: 1GB (FP16)
- Token buffer: 100MB (for 1M tokens × 8192 × 2 bytes)
- Communication buffer: 256MB
- Scratch space: 512MB
- Total per GPU: ~2GB (leaving 78GB for other uses on H100)