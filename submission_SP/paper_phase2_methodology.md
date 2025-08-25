# Phase 2: Methodology Extraction

## Abstract (Retained from Original)
We present a novel parallelization strategy for Multi-Head Attention (MHA) in large-scale transformer models that combines Ring Attention with sequence parallelism. Our approach leverages the communication-efficient properties of the ring topology to distribute attention computation across devices, while sequence parallelism reduces memory footprint by splitting input sequences across workers. This design minimizes all-to-all communication overhead, enhances scalability for extremely long sequences, and enables efficient utilization of distributed hardware resources. Experimental analysis indicates that the proposed method achieves substantial throughput improvements compared to conventional data- and tensor-parallel approaches, particularly in scenarios with high sequence length and large model size.

## Detailed Methodology

### 1. Notation and Problem Setup
- **Input**: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ where:
  - $B$: batch size
  - $L$: sequence length
  - $d_{\text{model}}$: model's hidden size
- **MHA Structure**: $H$ attention heads, each of dimension $d_h = d_{\text{model}} / H$
- **Attention Computation**: $\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_h}} \right) V$
- **Projections**: $Q = X W_Q$, $K = X W_K$, $V = X W_V$ with $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_h}$
- **Distributed Setup**: $P$ devices $\{D_0, D_1, \dots, D_{P-1}\}$
- **Objective**: Compute MHA in parallel with minimal communication overhead and reduced memory footprint

### 2. Sequence Parallelism
- **Data Splitting**: Sequence dimension $L$ is split across devices
- **Local Storage**: Each device $D_p$ stores $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$
- **Memory Reduction**: Activation memory reduced by factor of $P$
- **Challenge**: Self-attention requires all keys $K$ and values $V$ across entire sequence, creating communication bottleneck

### 3. Ring Attention Algorithm
- **Topology**: Devices arranged in logical ring
- **Stages**: $P$ sequential stages (0 to $P-1$)
- **Process**:
  1. **Initialization**: Each device computes local $Q^{(p)}, K^{(p)}, V^{(p)}$ from $X^{(p)}$
  2. **Ring Communication** (at stage $t$, $0 \leq t < P$):
     - Each device computes partial attention between local $Q^{(p)}$ and current $K^{(\text{src})}, V^{(\text{src})}$
     - Source index: $\text{src} \leftarrow (p - t) \bmod P$
     - Pass $K, V$ tensors to next device in ring
     - Accumulate partial attention results over stages
  3. **Aggregation**: After $P$ stages, each device has computed attention outputs for local queries using all keys and values

### 4. Combined Ring Attention + Sequence Parallelism
- **Integration**:
  - Sequence parallelism: Defines data placement (each device stores sequence slice)
  - Ring Attention: Defines communication order (sequential peer-to-peer exchanges)
- **Pseudocode**:
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
- **Naïve All-Gather**: Each device exchanges $\mathcal{O}(L d_{\text{model}})$ per step
- **Ring Attention**: Each device exchanges $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per stage, with $P$ stages
- **Total Volume**: Same as all-gather but with lower peak bandwidth and better overlap between communication and computation
- **Memory Cost**: Activation memory drops from $\mathcal{O}(L d_{\text{model}})$ to $\mathcal{O}(\frac{L}{P} d_{\text{model}})$

### 6. Implementation Details
- **Communication Primitives**: NCCL's `send/recv` or MPI point-to-point operations
- **Overlap Strategy**: Computation of attention for one block overlaps with asynchronous communication of next $K, V$ block
- **Precision**: Mixed-precision (`fp16` or `bf16`) for $Q, K, V$ to reduce bandwidth
- **Optimization**: Fused kernels for projection and softmax with communication hooks to reduce kernel launch overhead
- **Scalability**: Performance benefits grow with $L$ and $P$, particularly for $L > 16$k tokens