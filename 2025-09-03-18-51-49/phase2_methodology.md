# Phase 2: Methodology Extraction

## 1. Problem Setup and Notation

### Input Dimensions
- Input sequence: X ∈ ℝ^(B×L×d_model)
- B: batch size
- L: sequence length
- d_model: model's hidden size
- H: number of attention heads
- d_h = d_model/H: dimension per head

### Model Parameters
- W_Q, W_K, W_V ∈ ℝ^(d_model×d_h): projection matrices
- P: number of distributed devices {D_0, D_1, ..., D_{P-1}}

### Attention Computation
For single head: Attn(Q, K, V) = softmax(QK^T/√d_h)V
Where: Q = XW_Q, K = XW_K, V = XW_V

## 2. Sequence Parallelism Implementation

### Data Distribution
- Sequence dimension L split across P devices
- Each device D_p stores: X^(p) ∈ ℝ^(B×(L/P)×d_model)
- Memory reduction: from O(L·d_model) to O((L/P)·d_model)

### Challenge
- Self-attention requires all K and V across entire sequence
- Naïve approach requires all-gather operation (costly for large L)

## 3. Ring Attention Algorithm

### Ring Topology Setup
- Devices arranged in logical ring
- P stages of computation (0 ≤ t < P)

### Algorithm Steps

#### Stage 1: Initialization
Each device D_p computes from local X^(p):
- Q^(p), K^(p), V^(p)

#### Stage 2: Ring Communication (P stages)
At each stage t:
1. Compute partial attention between local Q^(p) and current KV_block
2. Pass KV_block to next device in ring
3. Receive KV_block from previous device
4. Accumulate partial results

#### Stage 3: Aggregation
After P stages, each device has computed attention outputs for its local queries using all keys and values across sequence

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

## 4. Combined RA+SP Integration

### Data Placement Strategy
- Sequence parallelism defines: each device stores L/P tokens
- Ring Attention defines: communication order via ring topology

### Communication Pattern
- Replaces all-gather with sequential peer-to-peer exchanges
- Each device sends/receives one block per stage
- Total communication volume same but peak bandwidth reduced

## 5. Implementation Details

### Technical Specifications
- **Communication Primitives**: NCCL send/recv or MPI point-to-point
- **Overlap Strategy**: Computation overlaps with asynchronous communication
- **Precision**: Mixed-precision (fp16 or bf16) for Q, K, V tensors
- **Kernel Optimization**: Fused projection and softmax with communication hooks
- **Scalability Threshold**: Benefits grow with L and P, especially L > 16k tokens

### Memory Management
- Activation memory: O((L/P)·d_model) per device
- KV cache: Distributed across devices via ring communication
- No full-sequence duplication on each worker

### Communication Complexity Analysis
- **Naïve All-Gather**: O(L·d_model) per device per step
- **Ring Attention**: O((L/P)·d_model) per stage × P stages = same total volume
- **Peak Bandwidth**: Significantly reduced due to sequential exchanges
- **Latency**: Better overlap between communication and computation