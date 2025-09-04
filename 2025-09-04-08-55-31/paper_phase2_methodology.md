# Phase 2: Methodology Extraction

## 1. Notation and Problem Setup

**Input Dimensions:**
- Input sequence: X ∈ ℝ^(B×L×d_model)
- B: batch size
- L: sequence length
- d_model: model's hidden size
- H: number of attention heads
- d_h = d_model/H: dimension per head

**Attention Computation for Single Head:**
```
Attn(Q, K, V) = softmax(QK^T/√d_h)V
```

**Weight Matrices:**
- Q = XW_Q, K = XW_K, V = XW_V
- W_Q, W_K, W_V ∈ ℝ^(d_model×d_h)

**Distributed Setup:**
- P distributed devices: {D_0, D_1, ..., D_{P-1}}
- Objective: Compute MHA in parallel with minimal communication overhead and reduced memory footprint

## 2. Sequence Parallelism Implementation

**Data Partitioning:**
- Sequence dimension L split across P devices
- X = [X^(0), X^(1), ..., X^(P-1)]
- Each device D_p stores: X^(p) ∈ ℝ^(B×L/P×d_model)
- Memory reduction: Activation memory per device drops from O(L×d_model) to O(L/P×d_model)

## 3. Ring Attention Algorithm

**Ring Topology:**
- Devices arranged in logical ring
- Communication proceeds in P sequential stages
- Each stage involves peer-to-peer exchanges

**Algorithm Steps:**

**Stage 1: Initialization**
- Each device computes local projections:
  - Q^(p), K^(p), V^(p) from X^(p)

**Stage 2: Ring Communication (P stages)**
For stage t (0 ≤ t < P):
1. Each device computes partial attention:
   - Using local Q^(p) and current KV_block
2. Pass KV_block to next device in ring:
   - src_idx = (p - t) mod P
3. Accumulate partial results over stages

**Stage 3: Aggregation**
- After P stages, each device has complete attention output for its local queries
- Uses all keys and values across entire sequence

## 4. Combined Ring Attention + Sequence Parallelism

**Integration Strategy:**
- Sequence parallelism: Defines data placement (sequence split)
- Ring attention: Defines communication order (ring-based KV passing)

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

## 5. Communication Complexity Analysis

**Naïve All-Gather:**
- Each device exchanges O(L×d_model) per step
- Peak bandwidth requirement: High

**Ring Attention:**
- Each device exchanges O(L/P×d_model) per stage
- P stages total, same total volume but lower peak bandwidth
- Better overlap between communication and computation

**Memory Cost:**
- Sequence parallelism reduces activation memory by factor P
- From O(L×d_model) to O(L/P×d_model) per device

## 6. Implementation Details

**Topology:**
- NCCL's send/recv primitives or MPI point-to-point operations
- Logical ring topology over physical interconnect

**Overlap Optimization:**
- Computation of attention for one block overlaps with asynchronous communication of next KV block
- Non-blocking send/receive operations

**Precision:**
- Mixed-precision (fp16 or bf16) for Q, K, V tensors
- Reduces communication bandwidth requirements

**Fused Kernels:**
- Projection and softmax operations fused with communication hooks
- Reduces kernel launch overhead

**Scalability:**
- Performance benefits increase with L (sequence length) and P (number of devices)
- Particularly effective for L > 16k tokens

## 7. Mathematical Formulation

**Ring Communication Pattern:**
For device p at stage t:
- Receives KV block from device (p-1) mod P
- Computes attention: partial = softmax(Q_p K_src^T/√d_h)V_src
- Sends KV block to device (p+1) mod P
- Accumulates: output_p += partial

**Index Calculation:**
- Source index for stage t: src_idx = (p - t) mod P
- Ensures each device eventually processes all KV blocks

**Final Output:**
- Each device D_p produces: output^(p) ∈ ℝ^(B×L/P×d_model)
- Complete output: [output^(0), output^(1), ..., output^(P-1)]