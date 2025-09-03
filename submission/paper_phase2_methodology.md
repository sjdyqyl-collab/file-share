# Phase 2: Methodology Extraction

## Problem Setup and Notation

**Input Dimensions:**
- Batch size: B
- Sequence length: L  
- Model hidden size: d_model
- Input tensor: X ∈ ℝ^(B×L×d_model)

**MHA Configuration:**
- Number of attention heads: H
- Head dimension: d_h = d_model / H

**Attention Computation for Single Head:**
```
Attn(Q, K, V) = softmax(QK^T/√d_h) V
```
Where:
- Q = XW_Q, K = XW_K, V = XW_V
- W_Q, W_K, W_V ∈ ℝ^(d_model×d_h)

**Distributed Setup:**
- P distributed devices: {D_0, D_1, ..., D_(P-1)}
- Objective: Compute MHA with minimal communication overhead and reduced memory footprint

## Sequence Parallelism Implementation

**Data Splitting:**
- Sequence dimension L split across P devices
- Each device D_p stores: X^(p) ∈ ℝ^(B×(L/P)×d_model)
- Memory reduction factor: P (from O(L×d_model) to O((L/P)×d_model))

**Communication Challenge:**
- Self-attention requires all keys K and values V across entire sequence
- Naive approach requires all-gather operation (costly for large L)

## Ring Attention Algorithm

**Ring Topology:**
- Devices arranged in logical ring
- Communication proceeds in P sequential stages

**Algorithm Stages:**
1. **Initialization:** Each device computes local Q^(p), K^(p), V^(p) from X^(p)

2. **Ring Communication (for t = 0 to P-1):**
   - Compute partial attention between local Q^(p) and current K^(src), V^(src)
   - Pass K, V tensors to next device in ring
   - Source index: src = (p - t) mod P
   - Accumulate partial attention results

3. **Aggregation:** After P stages, each device has computed attention outputs for local queries using all keys/values

## Combined Ring Attention + Sequence Parallelism

**Integration Strategy:**
- Sequence parallelism: Defines data placement (each device stores sequence slice)
- Ring attention: Defines communication order (sequential peer-to-peer exchanges)

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

## Communication Complexity Analysis

**Naive All-Gather:**
- Each device exchanges O(L×d_model) per step

**Ring Attention:**
- Each device exchanges O((L/P)×d_model) per stage
- P stages total, same volume but lower peak bandwidth
- Better overlap between communication and computation

**Memory Cost:**
- Activation memory per device: O((L/P)×d_model) (reduced from O(L×d_model))

## Implementation Details

**Technical Specifications:**
- **Topology:** NCCL send/recv primitives or MPI point-to-point operations
- **Overlap:** Attention computation overlaps with asynchronous K,V block communication
- **Precision:** Mixed-precision (fp16 or bf16) for Q,K,V tensors
- **Fused Kernels:** Projection and softmax fused with communication hooks
- **Scalability:** Benefits grow with L and P, especially for L > 16k tokens