# Phase 2: Methodology - Ring Attention + Sequence Parallelism

## 1. Notation and Problem Setup

### Input Dimensions
- Input sequence: X ∈ ℝ^(B×L×d_model)
  - B: batch size
  - L: sequence length  
  - d_model: model's hidden size
- H: number of attention heads
- d_h: dimension per head = d_model / H
- P: number of distributed devices {D_0, D_1, ..., D_{P-1}}

### MHA Computation
For single attention head:
```
Attn(Q, K, V) = softmax(QK^T/√d_h)V
```
Where:
- Q = XW_Q, K = XW_K, V = XW_V
- W_Q, W_K, W_V ∈ ℝ^(d_model×d_h)

## 2. Sequence Parallelism Implementation

### Data Partitioning
Sequence dimension L split across P devices:
```
X = [X^(0), X^(1), ..., X^(P-1)]
```
Where X^(p) ∈ ℝ^(B×(L/P)×d_model) resides on device D_p

### Memory Reduction
- Activation memory per device: O(L×d_model) → O((L/P)×d_model)
- Memory reduction factor: P

### Communication Challenge
- Self-attention requires all keys K and values V across entire sequence
- Naive approach: all-gather operation (costly for large L)

## 3. Ring Attention Algorithm

### Ring Topology Structure
- Devices arranged in logical ring
- Sequential peer-to-peer exchanges
- P stages total

### Algorithm Steps

#### Stage 1: Initialization
Each device computes local projections:
```
Q^(p), K^(p), V^(p) = Project(X^(p))
```

#### Stage 2: Ring Communication (P stages)
For t = 0 to P-1:
1. Compute partial attention:
   ```
   partial = Attention(Q^(p), KV_block)
   output_p += partial
   ```
2. Pass K,V tensors to next device in ring:
   ```
   src_idx = (p - t) mod P
   send KV_block to next device
   receive KV_block from previous device
   ```

#### Stage 3: Aggregation
After P stages, each device has computed attention outputs for its local queries using all keys and values across the sequence.

### Pseudocode Implementation
```
for p in parallel on devices:
    Q_p, K_p, V_p = Project(X_p)
    output_p = 0
    KV_block = (K_p, V_p)
    for t in 0..P-1:
        src_idx = (p - t) mod P
        partial = Attention(Q_p, KV_block)
        output_p += partial
        send KV_block to next device in ring
        receive KV_block from previous device
```

## 4. Combined RA+SP Strategy

### Integration Approach
- **Sequence Parallelism**: Defines data placement (sequence split)
- **Ring Attention**: Defines communication order (ring-based exchange)

### Communication Pattern
- Replaces all-gather with sequential peer-to-peer exchanges
- Each device sends/receives one block per stage
- Total communication volume: same as all-gather, but lower peak bandwidth

## 5. Communication Complexity Analysis

### Naive All-Gather
- Each device exchanges: O(L×d_model) per step
- Peak bandwidth: High (all-to-all pattern)

### Ring Attention
- Each device exchanges: O((L/P)×d_model) per stage
- P stages total
- Same total volume but lower peak bandwidth
- Better overlap between communication and computation

### Memory Scaling
- Sequence parallelism reduces activation memory from O(L×d_model) to O((L/P)×d_model)

## 6. Implementation Details

### Hardware Requirements
- NCCL send/recv primitives or MPI point-to-point operations
- Support for asynchronous communication
- Mixed-precision support (fp16/bf16)

### Optimization Techniques
- **Overlap**: Computation overlaps with asynchronous communication
- **Precision**: fp16/bf16 for Q,K,V to reduce bandwidth
- **Fused Kernels**: Projection and softmax fused with communication hooks
- **Scalability**: Benefits grow with L and P, especially L > 16k tokens

### Performance Characteristics
- Scales efficiently with sequence length and device count
- Particularly effective for memory-constrained environments
- Reduces synchronization overhead compared to global communication patterns