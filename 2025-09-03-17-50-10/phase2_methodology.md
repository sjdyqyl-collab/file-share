# Phase 2: Methodology Extraction

## 1. Notation and Problem Setup

**Input**: Transformer layer with MHA on input sequence
- $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$
- $B$: batch size
- $L$: sequence length  
- $d_{\text{model}}$: model hidden size

**MHA Structure**:
- $H$ attention heads
- $d_h = d_{\text{model}} / H$ (dimension per head)
- Single head attention: $\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_h}} \right) V$
- Projections: $Q = X W_Q, K = X W_K, V = X W_V$ with $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_h}$

**Distributed Setup**:
- $P$ devices $\{D_0, D_1, \dots, D_{P-1}\}$
- Objective: Compute MHA in parallel with minimal communication overhead and reduced memory footprint

## 2. Sequence Parallelism

**Data Partitioning**:
- Sequence dimension $L$ split across $P$ devices
- $X = [X^{(0)}, X^{(1)}, \dots, X^{(P-1)}]$ where $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$ on device $D_p$
- **Memory Reduction**: Activation memory reduced by factor of $P$

**Challenge**: Self-attention requires all keys $K$ and values $V$ across entire sequence, creating communication bottleneck

## 3. Ring Attention

**Topology**: Devices arranged in logical ring with sequential peer-to-peer exchanges

**Algorithm**: $P$ stages (0 ≤ t < P)

1. **Initialization**:
   - Each device computes local $Q^{(p)}, K^{(p)}, V^{(p)}$ from $X^{(p)}$

2. **Ring Communication** (stage t):
   - Compute partial attention between local $Q^{(p)}$ and current $K^{(\text{src})}, V^{(\text{src})}$
   - Pass $K, V$ tensors to next device in ring
   - Source index: $\text{src} \leftarrow (p - t) \bmod P$
   - Accumulate partial attention results over stages

3. **Aggregation**:
   - After $P$ stages, each device has computed attention outputs for local queries using all keys/values

## 4. Combined Ring Attention + Sequence Parallelism

**Integration**:
- Sequence parallelism: Defines data placement (sequence split across devices)
- Ring Attention: Defines communication order (sequential exchanges vs all-gather)

**Pseudocode**:
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

**Naïve All-Gather**:
- Each device exchanges $\mathcal{O}(L d_{\text{model}})$ per step

**Ring Attention**:
- Each device exchanges $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per stage
- $P$ stages total → same total volume but lower peak bandwidth
- Better overlap between communication and computation

**Memory Cost**:
- Activation memory per device: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ vs $\mathcal{O}(L d_{\text{model}})$

## 6. Implementation Details

**Communication**:
- NCCL's `send/recv` primitives or MPI point-to-point operations
- Overlap: Attention computation for one block overlaps with async communication of next KV block

**Optimization**:
- Mixed-precision: `fp16` or `bf16` for Q, K, V to reduce bandwidth
- Fused kernels: Projection and softmax fused with communication hooks
- Scalability: Benefits grow with L and P, particularly for L > 16k tokens