# Phase 2: Methodology Extraction

## 1. Notation and Problem Setup

### Input Dimensions
- Input sequence: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$
- Batch size: $B$
- Sequence length: $L$
- Model hidden size: $d_{\text{model}}$
- Number of attention heads: $H$
- Head dimension: $d_h = d_{\text{model}} / H$

### Distributed Setup
- Number of devices: $P$
- Devices: $\{D_0, D_1, \dots, D_{P-1}\}$

### Attention Computation
For single head:
$$\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_h}} \right) V$$

Where:
- $Q = X W_Q$
- $K = X W_K$
- $V = X W_V$
- $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_h}$

## 2. Sequence Parallelism

### Data Partitioning
Sequence dimension $L$ is split across $P$ devices:
$$X = [X^{(0)}, X^{(1)}, \dots, X^{(P-1)}]$$

Where each device $D_p$ stores:
- $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$

### Memory Reduction
- Activation memory per device: $\mathcal{O}(\frac{L}{P} \cdot d_{\text{model}})$
- Reduction factor: $P$

## 3. Ring Attention Algorithm

### Ring Topology Structure
- Devices arranged in logical ring
- Peer-to-peer communication pattern
- $P$ stages of computation

### Algorithm Steps

#### Stage 1: Initialization
Each device computes local projections:
- $Q^{(p)}, K^{(p)}, V^{(p)}$ from $X^{(p)}$

#### Stage 2: Ring Communication (P stages)
For each stage $t$ ($0 \leq t < P$):
1. Compute partial attention between local $Q^{(p)}$ and current $K^{(\text{src})}, V^{(\text{src})}$
2. Source device index: $\text{src} \leftarrow (p - t) \bmod P$
3. Pass $K, V$ tensors to next device in ring
4. Accumulate partial attention results

#### Stage 3: Aggregation
After $P$ stages:
- Each device has computed attention outputs for its local queries using all keys and values across the sequence

### Pseudocode
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
- Each device exchanges: $\mathcal{O}(L \cdot d_{\text{model}})$ per step
- Peak bandwidth requirement: High

### Ring Attention Approach
- Each device exchanges: $\mathcal{O}(\frac{L}{P} \cdot d_{\text{model}})$ per stage
- Number of stages: $P$
- Total volume: Same as all-gather but **lower peak bandwidth**
- Better overlap between communication and computation

## 5. Implementation Details

### Communication Primitives
- NCCL's `send/recv` primitives
- MPI point-to-point operations

### Optimization Techniques
- **Overlap**: Computation of attention for one block overlaps with asynchronous communication of next $K, V$ block
- **Precision**: Mixed-precision (fp16 or bf16) for $Q, K, V$ to reduce bandwidth
- **Fused Kernels**: Projection and softmax fused with communication hooks
- **Scalability**: Performance benefits grow with $L$ and $P$, particularly for $L > 16\text{k}$ tokens

### Memory Requirements
- **Activation memory**: $\mathcal{O}(\frac{L}{P} \cdot d_{\text{model}})$ per device
- **Parameter storage**: Unchanged from baseline
- **Communication buffers**: $\mathcal{O}(\frac{L}{P} \cdot d_{\text{model}})$ per device