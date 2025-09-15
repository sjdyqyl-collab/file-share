# Phase 2: Methodology Extraction

## Mathematical Notation
- Input sequence: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ where:
  - $B$ = batch size
  - $L$ = sequence length
  - $d_{\text{model}}$ = model's hidden size
- $H$ attention heads, each with dimension $d_h = d_{\text{model}} / H$
- $P$ distributed devices $\{D_0, D_1, \dots, D_{P-1}\}$

## Multi-Head Attention Computation
For a single head:
$$\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_h}} \right) V$$

Where:
- $Q = X W_Q$
- $K = X W_K$
- $V = X W_V$
- $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_h}$

## 1. Sequence Parallelism Implementation

### Data Distribution
- Sequence dimension $L$ is split across $P$ devices:
- $X = [X^{(0)}, X^{(1)}, \dots, X^{(P-1)}]$
- Each device $D_p$ stores $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$
- **Memory reduction**: Activation memory decreases from $\mathcal{O}(L d_{\text{model}})$ to $\mathcal{O}(\frac{L}{P} d_{\text{model}})$

### Challenge
- Self-attention requires all keys $K$ and values $V$ across entire sequence
- Naïve approach requires all-gather operation (costly for large $L$)

## 2. Ring Attention Algorithm

### Ring Topology Structure
- Devices arranged in logical ring
- Partial $K$ and $V$ blocks passed in fixed order
- $P$ stages total

### Algorithm Stages

#### Stage 1: Initialization
Each device computes local projections:
- $Q^{(p)}, K^{(p)}, V^{(p)}$ from local $X^{(p)}$

#### Stage 2: Ring Communication (for $t = 0$ to $P-1$)
1. **Computation**: Each device computes partial attention between local $Q^{(p)}$ and current $K^{(\text{src})}, V^{(\text{src})}$
2. **Communication**: Pass $K, V$ tensors to next device in ring
3. **Source calculation**: $\text{src} \leftarrow (p - t) \bmod P$
4. **Accumulation**: Accumulate partial attention results over stages

#### Stage 3: Aggregation
After $P$ stages, each device has computed attention outputs for its local queries using all keys and values across the sequence

## 3. Combined Ring Attention + Sequence Parallelism

### Integration Strategy
- **Sequence parallelism**: Defines data placement (each device stores sequence slice)
- **Ring Attention**: Defines communication order (sequential block exchange instead of all-gather)

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

## 4. Communication Complexity Analysis

### Naïve All-Gather Approach
- Each device exchanges $\mathcal{O}(L d_{\text{model}})$ per step
- Peak bandwidth requirement: $\mathcal{O}(L d_{\text{model}})$

### Ring Attention Approach
- Each device exchanges $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per stage
- $P$ stages total
- **Same total volume** but **lower peak bandwidth**
- Better overlap between communication and computation

## 5. Implementation Details

### Technical Specifications
- **Topology**: NCCL's `send/recv` primitives or MPI point-to-point operations
- **Overlap**: Computation for one block overlaps with asynchronous communication of next $K, V$ block
- **Precision**: Mixed-precision (`fp16` or `bf16`) for $Q, K, V$ tensors
- **Fused Kernels**: Projection and softmax operations fused with communication hooks
- **Scalability**: Benefits increase with $L$ and $P$, especially for $L > 16\text{k}$ tokens

### Memory Optimization
- Activation memory per device: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$
- No full-sequence duplication across devices
- Reduced memory fragmentation through sequential processing