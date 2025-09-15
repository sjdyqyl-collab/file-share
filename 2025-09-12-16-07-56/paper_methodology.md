# Phase Two: Detailed Methodology

## Method Overview
The proposed method maximizes expert-level parallelism in large-scale MoE models by enforcing at most one expert per GPU and distributing experts across nodes. This shifts optimization focus from communication reduction to maximizing compute concurrency.

## Detailed Expert Placement Strategy

### 2.1 Single-Expert-Per-GPU Deployment
**Constraint**: Each GPU hosts at most one expert

**Mathematical Formulation**:
- Let E = total number of experts per MoE layer (E = 16 in experiments)
- Let G = total number of available GPUs
- Let L = number of MoE layers (L = 4 in experiments)
- Total experts = E × L = 64

**Placement Rules**:
1. If E ≤ G: Each expert assigned to distinct GPU
2. If E > G: Experts replicated across GPUs with memory balancing
3. Each GPU must have sufficient memory for single expert computation

### 2.2 Cross-Node Distribution Algorithm
**Input**: Cluster topology graph T = (N, L) where N = nodes, L = links
**Output**: Expert-to-GPU mapping M: Expert → (Node, GPU_ID)

**Constraints**:
- Minimize max(tokens_sent_across_any_link)
- Ensure one-expert-per-GPU rule
- Balance memory usage across nodes

**Algorithm Steps**:
1. Calculate bandwidth matrix B[i,j] between nodes i and j
2. Estimate token routing probabilities P[expert] based on gating
3. Apply greedy placement minimizing communication cost
4. Verify memory constraints per GPU

## Routing and Load Balancing Details

### 3.1 Gating Mechanism Implementation
**Top-K Routing**: K = 2 (standard MoE practice)
**Gating Function**: softmax(W_gate * x) where W_gate ∈ ℝ^(E×d_model)
**Selection**: Top-2 experts per token based on gating scores

### 3.2 Token Sharding Implementation
**Token Representation**: Each token is a vector of dimension 8192
**Batch Processing**: 1024 sequences × 10000 tokens = 10,240,000 tokens

**Sharding Steps**:
1. For each token, determine destination experts via gating
2. Group tokens by destination expert ID
3. Create token batches for each expert
4. Asynchronously send batches to expert locations

### 3.3 Load Balancing Algorithm
**Dynamic Adjustment**:
1. Monitor token counts per expert over sliding window
2. Calculate load imbalance ratio = max(tokens_per_expert) / avg(tokens_per_expert)
3. If ratio > threshold (1.5), adjust gating probabilities:
   - Reduce probability for overloaded experts
   - Increase probability for underloaded experts
4. Apply exponential smoothing to prevent oscillation

## Communication Overlap and Scheduling

### 4.1 Overlapping Compute and Communication
**CUDA Streams Configuration**:
- Stream 0: Expert computation
- Stream 1: Token data transfer
- Stream 2: Gradient synchronization (if training)

**Overlap Pattern**:
1. While GPU i processes batch b of tokens
2. Simultaneously receive batch b+1 for expert on GPU i
3. Send results of batch b-1 to next layer

### 4.2 Pipeline Scheduling for Multi-Layer MoE
**Layer-wise Pipeline**:
- Each MoE layer = pipeline stage
- Token routing between layers happens immediately
- No waiting for full batch completion

**Scheduling Algorithm**:
```
for layer in 1..L:
    for expert in 1..E:
        if tokens_available(expert, layer):
            start_computation(expert, layer)
        if computation_complete(expert, layer-1):
            route_tokens_to_next_layer(expert, layer)
```

## Memory and Model Parallelism Integration

### 5.1 Tensor Parallelism within Expert
**When to Apply**: When single expert FFN exceeds GPU memory
**Configuration**: TP=2 (split expert across 2 GPUs)
**Split Strategy**: Column-parallel for first linear, Row-parallel for second linear

### 5.2 Data Parallelism Integration
**DP Degree**: Determined by available replicas
**Synchronization**: All-reduce across DP replicas after each step
**Expert Consistency**: Ensure same expert weights across replicas

## Implementation Parameters

### Model Dimensions
- Token dimension: 8192
- MLP hidden size: 32768
- MHA: 16 heads × 512 dim/head = 8192 total
- Number of layers: 4
- Experts per layer: 16
- Total experts: 64

### Batch Configuration
- Sequences per batch: 1024
- Tokens per sequence: 10000
- Total tokens per batch: 10,240,000
- Precision: FP16

### GPU Requirements
- Type: H100
- Memory per GPU: Must fit single expert + activations
- Network: NVLink/InfiniBand for inter-node communication
- Minimum GPUs: 64 (for 64 experts)
- Optimal GPUs: 64 (one expert per GPU)

### Communication Parameters
- NCCL for collective operations
- MPI for cross-node coordination
- CUDA streams for overlap
- Token batch size: Dynamic based on expert load
- Asynchronous communication enabled

## Critical Implementation Notes

### Expert Memory Calculation
For single expert on GPU:
- Expert parameters: 2 × (8192 × 32768 + 32768 × 8192) = 1.07GB in FP16
- Activations: Depends on batch size and sequence length
- Total memory per expert: ~2-4GB (leaves room for other components)

### Network Bandwidth Requirements
- Inter-node bandwidth: ≥ 50 GB/s (InfiniBand)
- Intra-node bandwidth: ≥ 300 GB/s (NVLink)
- Latency: < 5 μs for optimal overlap

### Load Balancing Thresholds
- Imbalance ratio threshold: 1.5
- Adjustment window: 1000 tokens
- Smoothing factor: 0.9