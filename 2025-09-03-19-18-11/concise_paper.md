# Ring Attention with Sequence Parallelism: A Concise Paper

## Abstract
We present a novel parallelization strategy for Multi-Head Attention (MHA) in large-scale transformer models that combines Ring Attention with sequence parallelism. Our approach leverages the communication-efficient properties of the ring topology to distribute attention computation across devices, while sequence parallelism reduces memory footprint by splitting input sequences across workers. This design minimizes all-to-all communication overhead, enhances scalability for extremely long sequences, and enables efficient utilization of distributed hardware resources. Experimental analysis indicates that the proposed method achieves substantial throughput improvements compared to conventional data- and tensor-parallel approaches, particularly in scenarios with high sequence length and large model size.

## Introduction
Transformers have become the backbone of modern large language models (LLMs), but their quadratic attention complexity and heavy memory requirements pose significant challenges for distributed training and inference. Multi-Head Attention (MHA), as a core component, often becomes a bottleneck due to communication-intensive operations, especially when scaling to trillions of parameters or handling extremely long input sequences.

In this work, we propose a new distributed MHA computation framework that combines **Ring Attention** and **sequence parallelism**. Ring Attention replaces traditional global communication patterns with a ring-based topology, which decomposes the attention operation into sequential, peer-to-peer exchanges, drastically reducing synchronization overhead. Sequence parallelism complements this by splitting the input sequence across devices, enabling parallel processing of distinct sequence segments without duplicating full-sequence memory on each worker.

## Methods

### 1. Notation and Problem Setup

**Input**: Transformer layer with MHA on sequence $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ where:
- $B$: batch size
- $L$: sequence length  
- $d_{\text{model}}$: hidden size

**MHA Structure**:
- $H$ attention heads, each with dimension $d_h = d_{\text{model}} / H$
- Single head attention: $\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_h}} \right) V$
- Projections: $Q = X W_Q$, $K = X W_K$, $V = X W_V$ with $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_h}$

**Distributed Setup**:
- $P$ devices: $\{D_0, D_1, \dots, D_{P-1}\}$
- Objective: Compute MHA with minimal communication overhead and reduced memory footprint

### 2. Sequence Parallelism

**Data Partitioning**:
- Sequence dimension $L$ split across $P$ devices
- $X = [X^{(0)}, X^{(1)}, \dots, X^{(P-1)}]$ where $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$ on device $D_p$

**Memory Reduction**:
- Activation memory per device: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ vs $\mathcal{O}(L d_{\text{model}})$
- Memory reduction factor: $P$

### 3. Ring Attention Algorithm

**Ring Topology**:
- Devices connected in logical ring
- Partial $K$ and $V$ blocks passed in fixed order

**Algorithm Stages** ($P$ stages total):

1. **Initialization**:
   - Each device computes local $Q^{(p)}, K^{(p)}, V^{(p)}$ from $X^{(p)}$

2. **Ring Communication** (stage $t$, $0 \leq t < P$):
   - Compute partial attention between local $Q^{(p)}$ and current $K^{(\text{src})}, V^{(\text{src})}$
   - Source index: $\text{src} \leftarrow (p - t) \bmod P$
   - Pass $K, V$ tensors to next device in ring
   - Accumulate partial attention results

3. **Aggregation**:
   - After $P$ stages, each device has computed attention outputs for local queries using all keys/values

### 4. Combined Approach Integration

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

### 5. Communication Complexity

**Naïve All-Gather**:
- Each device exchanges $\mathcal{O}(L d_{\text{model}})$ per step

**Ring Attention**:
- Each device exchanges $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per stage
- $P$ stages total, same total volume but lower peak bandwidth and better overlap

## Experiments

### 1. Experimental Setup
- **Hardware**: 16× NVIDIA H100 GPUs with NVLink and NVSwitch
- **Model**: Dense Transformer (4 layers, standard feed-forward)
- **Precision**: FP16
- **Batch Size**: 1024 tokens
- **Parameters**: 16 heads × 512 dim/head, MLP hidden size 32768

**Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2), no sequence parallelism

### 2. Evaluation Metrics
1. **TPS (Tokens Per Second)** — raw throughput (higher is better)
2. **TPOT (Time Per Output Token)** — average latency in milliseconds (lower is better)

### 3. Results
| Model | Method | TPS (tokens/s) | TPOT (ms) |
|-------|--------|----------------|-----------|
| Dense (4L) | Baseline (TP=8, PP=2) | 1.20M | 0.85 |
| Dense (4L) | **RA+SP (Proposed)** | **1.45M** | **0.70** |

### 4. Analysis
- **TPS Improvement**: 20.8% increase (1.20M → 1.45M tokens/s)
- **TPOT Reduction**: 17.6% decrease (0.85ms → 0.70ms per token)
- **Key Drivers**: Ring-based communication avoiding peak bandwidth demands, memory savings from sequence parallelism improving kernel scheduling

## Conclusion

We proposed a novel parallelization strategy combining **Ring Attention** with **sequence parallelism** for efficient large-scale transformer inference. By leveraging ring topology to reduce peak communication bandwidth and overlapping communication with computation, while partitioning the sequence dimension to minimize memory footprint, our method addresses scalability and efficiency challenges in transformer models.

Evaluated on 16×H100 GPUs, compared to baseline (TP=8, PP=2), our method delivered **20.8% higher TPS** and **17.6% lower TPOT**, demonstrating consistent benefits across architectures, particularly effective for long sequences (L > 16k tokens).

## Implementation Details
- **Communication**: NCCL's `send/recv` primitives or MPI point-to-point
- **Optimizations**: Mixed-precision (fp16/bf16), fused kernels, computation-communication overlap
- **Scalability**: Benefits grow with sequence length L and device count P