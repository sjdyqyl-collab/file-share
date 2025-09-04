# Phase 2: Methodology Extraction

## Mathematical Notation and Setup

### Input Specifications
- Input sequence: X ∈ ℝ^(B×L×d_model)
- Batch size: B
- Sequence length: L
- Model hidden size: d_model
- Number of attention heads: H
- Head dimension: d_h = d_model / H
- Number of distributed devices: P = {D_0, D_1, ..., D_{P-1}}

### Attention Computation
Single head attention:
```
Attn(Q, K, V) = softmax(QK^T/√d_h) V
```
Where:
- Q = XW_Q, K = XW_K, V = XW_V
- W_Q, W_K, W_V ∈ ℝ^(d_model×d_h)

## Sequence Parallelism

### Data Distribution
- Sequence dimension L split across P devices
- Each device D_p stores: X^(p) ∈ ℝ^(B×(L/P)×d_model)
- Memory reduction: Activation memory drops from O(L×d_model) to O((L/P)×d_model)

### Communication Challenge
- Self-attention requires all keys K and values V across entire sequence
- Naive approach requires all-gather operation: costly for large L

## Ring Attention Algorithm

### Ring Topology Structure
- Devices arranged in logical ring
- Partial K,V blocks passed sequentially between neighbors
- P stages for P devices

### Algorithm Steps

#### Stage 1: Initialization
```
Each device D_p:
    Q_p, K_p, V_p = Project(X_p)
    output_p = 0
    KV_block = (K_p, V_p)
```

#### Stage 2: Ring Communication (P stages)
For t = 0 to P-1:
```
Each device D_p:
    src_idx = (p - t) mod P
    partial = Attention(Q_p, KV_block)
    output_p += partial
    send KV_block to next device in ring
    receive KV_block from previous device
```

#### Stage 3: Final Output
After P stages, each device has:
- Computed attention outputs for local queries using all keys/values
- Distributed computation complete across ring

## Combined RA+SP Integration

### Data Placement Strategy
- Sequence parallelism: Each device stores L/P tokens
- Ring attention: Defines communication order for K,V blocks

### Communication Pattern
- Instead of all-gather: Each device sends/receives one block per stage
- Sequential peer-to-peer exchanges replace global communication

### Pseudocode Implementation
```python
# Parallel execution on P devices
for p in range(P):
    # Local computation
    Q_p, K_p, V_p = linear_projection(X_p)
    output_p = torch.zeros_like(Q_p)
    
    # Ring communication
    KV_block = (K_p, V_p)
    for t in range(P):
        src_idx = (p - t) % P
        partial = scaled_dot_product_attention(Q_p, KV_block[0], KV_block[1])
        output_p += partial
        
        # Async communication
        send_to_next(KV_block)
        KV_block = receive_from_prev()
```

## Communication Complexity Analysis

### Bandwidth Requirements
- Naive all-gather: O(L×d_model) per device per step
- Ring attention: O((L/P)×d_model) per stage × P stages = same total volume
- **Advantage**: Lower peak bandwidth, better computation-communication overlap

### Memory Scaling
- Sequence parallelism: Activation memory reduced by factor P
- No increase in parameter synchronization costs

## Implementation Specifications

### Hardware Requirements
- NCCL send/recv primitives or MPI point-to-point operations
- Support for asynchronous communication
- Mixed precision support (FP16/BF16)

### Optimization Techniques
- Computation overlap: Attention computation overlaps with async communication
- Fused kernels: Projection and softmax operations fused with communication hooks
- Precision: FP16/BF16 for Q,K,V to reduce bandwidth usage

### Scalability Parameters
- Performance benefits increase with:
  - Sequence length L (especially L > 16k tokens)
  - Number of devices P
- Optimal for memory-constrained, bandwidth-limited environments