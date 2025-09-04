# Methodology: Ring Attention + Sequence Parallelism

## 1. Notation and Problem Setup

**Input Configuration:**
- Input sequence: X ∈ ℝ^(B×L×d_model)
- B: batch size
- L: sequence length  
- d_model: model's hidden size
- H: number of attention heads
- d_h = d_model/H: dimension per head
- P: number of distributed devices {D_0, D_1, ..., D_{P-1}}

**MHA Computation:**
- Single head attention: Attn(Q, K, V) = softmax(QK^T/√d_h)V
- Q = XW_Q, K = XW_K, V = XW_V
- W_Q, W_K, W_V ∈ ℝ^(d_model×d_h)

## 2. Sequence Parallelism

**Data Distribution:**
- Sequence dimension L split across P devices
- X = [X^(0), X^(1), ..., X^(P-1)]
- Each device D_p stores X^(p) ∈ ℝ^(B×(L/P)×d_model)
- **Memory reduction:** Activation memory per device reduced from O(L×d_model) to O((L/P)×d_model)

**Challenge:** Self-attention requires all keys K and values V across entire sequence, creating communication bottleneck

## 3. Ring Attention Algorithm

**Ring Topology:** Devices arranged in logical ring with sequential peer-to-peer exchanges

**Algorithm Stages (P stages total):**

1. **Initialization:**
   - Each device computes local Q^(p), K^(p), V^(p) from X^(p)

2. **Ring Communication (for t = 0 to P-1):**
   - src_idx = (p - t) mod P
   - Compute partial attention between local Q^(p) and current K^(src), V^(src)
   - Accumulate partial results
   - Pass K, V tensors to next device in ring
   - Receive K, V tensors from previous device

3. **Final Aggregation:**
   - After P stages, each device has computed attention outputs for local queries using all keys/values

**Pseudocode:**
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

## 4. Communication Complexity Analysis

**Naïve All-Gather:**
- Each device exchanges O(L×d_model) per step
- Peak bandwidth: O(L×d_model)

**Ring Attention:**
- Each device exchanges O((L/P)×d_model) per stage
- P stages total, same total volume but lower peak bandwidth
- Better overlap between communication and computation

## 5. Implementation Details

**Technical Specifications:**
- **Topology:** NCCL send/recv primitives or MPI point-to-point operations
- **Overlap:** Attention computation for one block overlaps with async communication of next K,V block
- **Precision:** Mixed-precision (fp16 or bf16) for Q,K,V to reduce bandwidth
- **Fused Kernels:** Projection and softmax fused with communication hooks
- **Scalability:** Benefits increase with L and P, especially for L > 16k tokens

**Memory Optimization:**
- Activation memory reduced by factor of P
- Each device stores only L/P tokens
- No full-sequence duplication across devices