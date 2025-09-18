# Phase 2: Methodology Extraction

## Mathematical Notation and Problem Setup

### Input Representation
- Input sequence: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$
  - $B$: batch size
  - $L$: sequence length
  - $d_{\text{model}}$: model's hidden size
- $H$ attention heads, each with dimension $d_h = d_{\text{model}} / H$

### Multi-Head Attention Computation
For a single attention head:
- $Q = X W_Q$, $K = X W_K$, $V = X W_V$
- $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_h}$
- $\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_h}} \right) V$

### Distributed Setup
- $P$ distributed devices $\{D_0, D_1, \dots, D_{P-1}\}$
- Objective: compute MHA in parallel with minimal communication overhead and reduced memory footprint

## Sequence Parallelism

### Data Distribution
- Sequence dimension $L$ split across $P$ devices:
- $X = [X^{(0)}, X^{(1)}, \dots, X^{(P-1)}]$
- Each $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$ resides on device $D_p$
- **Memory reduction**: Activation memory per device drops from $\mathcal{O}(L d_{\text{model}})$ to $\mathcal{O}(\frac{L}{P} d_{\text{model}})$

### Communication Challenge
- Self-attention requires all keys $K$ and values $V$ across entire sequence
- Naïve approach requires all-gather operation (costly for large $L$)

## Ring Attention Algorithm

### Ring Topology Structure
- Devices arranged in logical ring
- Partial $K$ and $V$ blocks passed in fixed order
- $P$ stages total (0 ≤ t < P)

### Algorithm Steps

#### 1. Initialization
- Each device computes local:
  - $Q^{(p)}, K^{(p)}, V^{(p)}$ from local $X^{(p)}$

#### 2. Ring Communication (P stages)
At stage $t$ ($0 \leq t < P$):
- **Source index**: $\text{src} \leftarrow (p - t) \bmod P$
- **Computation**: Each device computes partial attention between local $Q^{(p)}$ and current $K^{(\text{src})}, V^{(\text{src})}$
- **Communication**: Pass $K, V$ tensors to next device in ring
- **Accumulation**: Accumulate partial attention results over stages

#### 3. Aggregation
- After $P$ stages, each device has computed attention outputs for local queries using all keys and values across sequence

## Combined Ring Attention + Sequence Parallelism

### Integration Strategy
- **Sequence parallelism**: Defines data placement (each device stores sequence slice)
- **Ring attention**: Defines communication order (ring-based instead of all-gather)

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

## Communication Complexity Analysis

### Naïve All-Gather Approach
- Each device exchanges $\mathcal{O}(L d_{\text{model}})$ per step
- High peak bandwidth requirement

### Ring Attention Approach
- Each device exchanges $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per stage
- $P$ stages total, same total volume but:
  - **Lower peak bandwidth**
  - **Better overlap** between communication and computation

## Implementation Details

### Technical Specifications
- **Topology**: NCCL's `send/recv` primitives or MPI point-to-point operations
- **Overlap**: Computation of attention for one block overlaps with asynchronous communication of next $K, V$ block
- **Precision**: Mixed-precision (`fp16` or `bf16`) for $Q, K, V$ tensors
- **Fused Kernels**: Projection and softmax operations fused with communication hooks
- **Scalability**: Performance benefits increase with $L$ and $P$, especially for $L > 16\text{k}$ tokens

### Memory Optimization
- Activation memory reduced by factor of $P$
- No duplication of full-sequence memory on each worker
- Efficient kernel scheduling due to reduced memory footprint