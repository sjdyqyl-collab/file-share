# Ring Attention with Sequence Parallelism: A Concise Version

**Abstract**
We present a novel parallelization strategy for Multi-Head Attention (MHA) in large-scale transformer models that combines Ring Attention with sequence parallelism. Our approach leverages the communication-efficient properties of the ring topology to distribute attention computation across devices, while sequence parallelism reduces memory footprint by splitting input sequences across workers. This design minimizes all-to-all communication overhead, enhances scalability for extremely long sequences, and enables efficient utilization of distributed hardware resources. Experimental analysis indicates that the proposed method achieves substantial throughput improvements compared to conventional data- and tensor-parallel approaches, particularly in scenarios with high sequence length and large model size.

## Methods

### Problem Setup
- **Input**: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ where $B$ is batch size, $L$ is sequence length, $d_{\text{model}}$ is hidden size
- **MHA**: $H$ attention heads, each dimension $d_h = d_{\text{model}} / H$
- **Distributed**: $P$ devices $\{D_0, D_1, \dots, D_{P-1}\}$

### Sequence Parallelism
Split sequence dimension $L$ across devices: each device $D_p$ stores $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$, reducing activation memory by factor $P$.

### Ring Attention Algorithm
1. **Initialization**: Each device computes local $Q^{(p)}, K^{(p)}, V^{(p)}$
2. **Ring Communication** (P stages):
   - Compute partial attention between local $Q^{(p)}$ and current $K^{(\text{src})}, V^{(\text{src})}$
   - Source: $\text{src} \leftarrow (p - t) \bmod P$
   - Pass $K, V$ to next device in ring
   - Accumulate results
3. **Final**: Each device has computed attention for local queries using all keys/values

### Combined Approach
```
for p in parallel on devices:
    Q_p, K_p, V_p = Project(X_p)
    output_p = 0
    KV_block = (K_p, V_p)
    for t in 0..P-1:
        src_idx = (p - t) mod P
        partial = Attention(Q_p, KV_block)
        output_p += partial
        send KV_block to next device
        receive KV_block from previous
```

### Implementation
- **Topology**: NCCL send/recv or MPI point-to-point
- **Overlap**: Computation overlaps with async communication
- **Precision**: Mixed-precision (fp16/bf16)
- **Memory**: Activation memory drops from $\mathcal{O}(L d_{\text{model}})$ to $\mathcal{O}(\frac{L}{P} d_{\text{model}})$

## Experiments

### Setup
- **Hardware**: 16×NVIDIA H100 GPUs, NVLink+NVSwitch
- **Models**: 
  - Dense Transformer: 4 layers
  - MoE: 4 layers, top-2 gating, 8 experts
- **Parameters**: FP16, batch=1024 tokens, 16 heads×512 dim, MLP=32768
- **Baseline**: TP=8, PP=2 (no sequence parallelism)

### Results
| Model | Method | TPS (M) | TPOT (ms) |
|-------|--------|---------|-----------|
| Dense | Baseline | 1.20 | 0.85 |
| Dense | RA+SP | **1.45** | **0.70** |
| MoE | Baseline | 0.95 | 1.05 |
| MoE | RA+SP | **1.18** | **0.82** |

### Key Findings
- Dense: 20.8% TPS↑, 17.6% TPOT↓
- MoE: 24.2% TPS↑, 21.9% TPOT↓
- Benefits scale with $L$ and $P$, especially $L > 16$k tokens

## Conclusion
Ring Attention with sequence parallelism provides efficient large-scale transformer inference by combining ring topology communication with sequence partitioning, achieving 20-25% throughput improvements and 17-27% latency reductions across architectures.