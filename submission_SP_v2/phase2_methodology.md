# Phase 2: Methodology Extraction

## Mathematical Formulation and Algorithms

### 1. Notation and Problem Setup
- **Input**: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ where:
  - $B$ = batch size
  - $L$ = sequence length
  - $d_{\text{model}}$ = model's hidden size
- **MHA Structure**: $H$ attention heads, each with dimension $d_h = d_{\text{model}} / H$
- **Projections**: $Q = X W_Q$, $K = X W_K$, $V = X W_V$ with $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_h}$
- **Single Head Attention**: $\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_h}} \right) V$
- **Devices**: $P$ distributed devices $\{D_0, D_1, \dots, D_{P-1}\}$

### 2. Sequence Parallelism Algorithm
- **Sequence Split**: $X = [X^{(0)}, X^{(1)}, \dots, X^{(P-1)}]$ where $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$ on device $D_p$
- **Memory Reduction**: Activation memory per device reduced by factor of $P$
- **Communication Challenge**: Requires all $K$ and $V$ across entire sequence for self-attention

### 3. Ring Attention Algorithm

#### Staged Process (P stages total):
**Stage 0 (Initialization):**
- Each device computes local $Q^{(p)}, K^{(p)}, V^{(p)}$ from local $X^{(p)}$

**Stage t (0 ≤ t < P):**
- Source device index: $\text{src} = (p - t) \bmod P$
- Each device computes partial attention: $\text{partial}_t = \text{Attention}(Q^{(p)}, K^{(\text{src})}, V^{(\text{src})})$
- Accumulate: $\text{output}^{(p)} += \text{partial}_t$
- Communication: Pass $(K^{(\text{src})}, V^{(\text{src})})$ to next device in ring
- Receive next $(K, V)$ block from previous device

### 4. Combined Algorithm Pseudocode
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

### 5. Communication Complexity Analysis
- **Naïve All-Gather**: Each device exchanges $\mathcal{O}(L \cdot d_{\text{model}})$ per step
- **Ring Attention**: Each device exchanges $\mathcal{O}(\frac{L}{P} \cdot d_{\text{model}})$ per stage, $P$ stages total
- **Peak Bandwidth**: Lower peak bandwidth due to sequential communication pattern
- **Memory**: Activation memory drops from $\mathcal{O}(L \cdot d_{\text{model}})$ to $\mathcal{O}(\frac{L}{P} \cdot d_{\text{model}})$

### 6. Implementation Specifications
- **Communication Primitives**: NCCL `send/recv` or MPI point-to-point operations
- **Overlap Strategy**: Computation of attention for current block overlaps with async communication of next $K,V$ block
- **Precision**: Mixed-precision (FP16 or BF16) for $Q,K,V$ tensors
- **Kernel Optimization**: Fused kernels for projection and softmax with communication hooks
- **Scalability Threshold**: Optimal for $L > 16\text{k}$ tokens

### 7. Model Architecture Details
- **Dense Transformer**: 4 layers, standard feed-forward
- **MoE Transformer**: 4 layers, top-2 gating, 8 experts, capacity factor 1.25
- **Fixed Parameters**:
  - Batch size: 1024 tokens
  - Number of heads: 16
  - Head dimension: 512
  - MLP hidden size: 32768
  - Precision: FP16
- **Expert Routing**: Performed locally to avoid communication for inactive experts