# Phase 2: Methodology Extraction

## Notation and Problem Setup

**Input Dimensions:**
- $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ where:
  - $B$ = batch size
  - $L$ = sequence length
  - $d_{\text{model}}$ = model's hidden size

**MHA Configuration:**
- $H$ attention heads
- Each head dimension: $d_h = d_{\text{model}} / H$
- Attention computation per head: $\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_h}} \right) V$

**Distributed Setup:**
- $P$ distributed devices $\{D_0, D_1, \dots, D_{P-1}\}$
- Objective: Compute MHA in parallel with minimal communication overhead and reduced memory footprint

## Sequence Parallelism Implementation

**Data Partitioning:**
- Sequence dimension $L$ split across $P$ devices
- Each device $D_p$ stores: $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$
- Memory reduction: Activation memory per device drops from $\mathcal{O}(L d_{\text{model}})$ to $\mathcal{O}(\frac{L}{P} d_{\text{model}})$

## Ring Attention Algorithm

**Topology:**
- Devices arranged in logical ring topology
- Communication proceeds in $P$ sequential stages

**Algorithm Stages:**

1. **Initialization:**
   - Each device computes local $Q^{(p)}, K^{(p)}, V^{(p)}$ from $X^{(p)}$

2. **Ring Communication (Stage $t$, $0 \leq t < P$):**
   - Each device computes partial attention between local $Q^{(p)}$ and current $K^{(\text{src})}, V^{(\text{src})}$
   - Source index calculation: $\text{src} \leftarrow (p - t) \bmod P$
   - Accumulate partial attention results over stages
   - Pass $K, V$ tensors to next device in ring

3. **Aggregation:**
   - After $P$ stages, each device has computed attention outputs for local queries using all keys and values across sequence

## Combined Ring Attention + Sequence Parallelism

**Integration Strategy:**
- Sequence parallelism defines data placement: each device stores only a slice of sequence
- Ring Attention defines communication order: sends/receives one block per stage instead of all-gather

**Pseudocode Implementation:**
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

**Naïve All-Gather:**
- Each device exchanges $\mathcal{O}(L d_{\text{model}})$ per step

**Ring Attention:**
- Each device exchanges $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per stage
- $P$ stages total, yielding same total volume but lower peak bandwidth
- Better overlap between communication and computation

## Implementation Details

**Communication Layer:**
- NCCL's `send/recv` primitives or MPI point-to-point operations
- Asynchronous communication for overlap with computation

**Optimization Techniques:**
- Mixed-precision (`fp16` or `bf16`) for $Q, K, V$ tensors
- Fused kernels for projection and softmax with communication hooks
- Computation of attention for one block overlaps with communication of next $K, V$ block

**Scalability Characteristics:**
- Performance benefits increase with sequence length $L$ and number of devices $P$
- Particularly effective for $L > 16\text{k}$ tokens
- Suitable for large-scale transformer deployments on distributed GPU clusters

## Memory Footprint

**Sequence Parallelism Memory Cost:**
- Activation memory per device: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$
- Compared to baseline: $\mathcal{O}(L d_{\text{model}})$
- Memory reduction factor: $P$ (number of devices)