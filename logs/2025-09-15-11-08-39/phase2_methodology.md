# Phase 2: Methodology Extraction

## 1. Notation and Problem Setup

### Input Specifications
- Input sequence: $X \in \mathbb{R}^{B \times L \times d_{model}}$
- Where: $B$ = batch size, $L$ = sequence length, $d_{model}$ = model hidden size
- MHA has $H$ attention heads, each of dimension $d_h = d_{model}/H$

### Attention Computation
For single head:
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V$$

Where:
- $Q = X W_Q$, $K = X W_K$, $V = X W_V$
- $W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_h}$

### Distributed Setup
- $P$ distributed devices $\{D_0, D_1, \dots, D_{P-1}\}$
- Objective: Compute MHA in parallel with minimal communication overhead and reduced memory footprint

## 2. Sequence Parallelism Implementation

### Data Partitioning
- Sequence dimension $L$ split across devices:
- $X = [X^{(0)}, X^{(1)}, \dots, X^{(P-1)}]$
- Each device stores: $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{model}}$

### Memory Reduction
- Activation memory reduced by factor of $P$
- From $\mathcal{O}(L \cdot d_{model})$ to $\mathcal{O}(\frac{L}{P} \cdot d_{model})$ per device

## 3. Ring Attention Algorithm

### Ring Topology Structure
- Devices arranged in logical ring
- Communication proceeds in $P$ stages

### Algorithm Steps

#### Stage 1: Initialization
- Each device computes local projections:
```
Q_p, K_p, V_p = Project(X_p)
```

#### Stage 2: Ring Communication (P stages)
For each stage $t$ ($0 \leq t < P$):
- Source index calculation: $\text{src} \leftarrow (p - t) \bmod P$
- Each device computes partial attention between local $Q^{(p)}$ and current $K^{(\text{src})}, V^{(\text{src})}$
- Accumulate partial attention results over stages
- Pass $K, V$ tensors to next device in ring

#### Stage 3: Aggregation
- After $P$ stages, each device has computed attention outputs for its local queries using all keys and values across sequence

### Communication Pattern
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

## 4. Combined Ring Attention + Sequence Parallelism

### Integration Strategy
- **Sequence parallelism**: Defines data placement (sequence slice per device)
- **Ring attention**: Defines communication order (ring-based K/V exchange)
- Eliminates need for expensive all-gather operations

### Communication Complexity Analysis

#### Naïve All-Gather Approach
- Each device exchanges $\mathcal{O}(L \cdot d_{model})$ per step

#### Ring Attention Approach
- Each device exchanges $\mathcal{O}(\frac{L}{P} \cdot d_{model})$ per stage
- Total $P$ stages → same total volume but lower peak bandwidth
- Better overlap between communication and computation

## 5. Implementation Details

### Technical Specifications
- **Topology**: Implemented over NCCL's `send/recv` primitives or MPI point-to-point operations
- **Overlap**: Computation of attention for one block overlaps with asynchronous communication of next $K, V$ block
- **Precision**: Mixed-precision (fp16 or bf16) for $Q, K, V$ to reduce bandwidth
- **Fused Kernels**: Projection and softmax fused with communication hooks to reduce kernel launch overhead
- **Scalability**: Performance benefits grow with $L$ and $P$, particularly for $L > 16k$ tokens

### Memory Layout
- Each device stores only $\frac{1}{P}$ of sequence
- K/V blocks passed in ring topology
- No full-sequence duplication across devices