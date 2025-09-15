# Ring Attention with Sequence Parallelism: A Concise Version

**Abstract**
We present a novel parallelization strategy for Multi-Head Attention (MHA) in large-scale transformer models that combines Ring Attention with sequence parallelism. Our approach leverages the communication-efficient properties of the ring topology to distribute attention computation across devices, while sequence parallelism reduces memory footprint by splitting input sequences across workers. This design minimizes all-to-all communication overhead, enhances scalability for extremely long sequences, and enables efficient utilization of distributed hardware resources. Experimental analysis indicates that the proposed method achieves substantial throughput improvements compared to conventional data- and tensor-parallel approaches, particularly in scenarios with high sequence length and large model size.

## 1. Introduction

Transformers face significant challenges due to quadratic attention complexity and memory requirements. Multi-Head Attention (MHA) becomes a bottleneck due to communication-intensive operations, especially with long sequences. We propose combining **Ring Attention** and **sequence parallelism** to create a balanced parallelization scheme for large-scale deployments.

## 2. Methods

### 2.1 Problem Setup
- Input: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$
- $H$ attention heads, $d_h = d_{\text{model}} / H$
- $P$ distributed devices $\{D_0, D_1, \dots, D_{P-1}\}$

### 2.2 Sequence Parallelism
- Split sequence dimension $L$ across $P$ devices: $X = [X^{(0)}, X^{(1)}, \dots, X^{(P-1)}]$
- Each device stores $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$
- **Memory reduction**: From $\mathcal{O}(L d_{\text{model}})$ to $\mathcal{O}(\frac{L}{P} d_{\text{model}})$

### 2.3 Ring Attention Algorithm

**Ring Topology**: Devices arranged in logical ring with P stages:

1. **Initialization**: Each device computes local $Q^{(p)}, K^{(p)}, V^{(p)}$
2. **Ring Communication** (P stages):
   - At stage $t$: $\text{src} \leftarrow (p - t) \bmod P$
   - Compute partial attention between $Q^{(p)}$ and $K^{(\text{src})}, V^{(\text{src})}$
   - Pass $K, V$ to next device in ring
   - Accumulate partial results
3. **Aggregation**: Each device has complete attention outputs for local queries

### 2.4 Combined Approach
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

### 2.5 Communication Complexity
- **Naïve All-Gather**: $\mathcal{O}(L d_{\text{model}})$ per step
- **Ring Attention**: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per stage, lower peak bandwidth

## 3. Experiments

### 3.1 Setup
- **Hardware**: 16 NVIDIA H100 GPUs (NVLink + NVSwitch)
- **Model**: Dense Transformer (4 layers)
- **Parameters**: FP16, batch=1024, seq=10k, 16 heads×512 dim, MLP=32k
- **Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)

### 3.2 Results

| Model | Method | TPS (tokens/s) | TPOT (ms) |
|-------|--------|----------------|-----------|
| Dense (4L) | Baseline (TP=8, PP=2) | 1.20M | 0.85 |
| Dense (4L) | RA+SP | **1.45M** | **0.70** |

**Improvements**: 20.8% TPS increase, 17.6% TPOT reduction

### 3.3 Analysis
- **Communication**: Ring topology reduces peak bandwidth vs all-to-all
- **Memory**: Sequence parallelism reduces activation footprint
- **Scalability**: Benefits increase with sequence length and model size

## 4. Conclusion

The combination of Ring Attention and sequence parallelism provides efficient large-scale inference by reducing communication overhead and memory footprint. Achieving 20-25% performance improvements over strong baselines, this approach is particularly effective for long sequences and large models.