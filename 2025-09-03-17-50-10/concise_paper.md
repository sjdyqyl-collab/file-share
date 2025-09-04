# Ring Attention with Sequence Parallelism: A Concise Paper

## Abstract
We present a novel parallelization strategy for Multi-Head Attention (MHA) in large-scale transformer models that combines Ring Attention with sequence parallelism. Our approach leverages the communication-efficient properties of the ring topology to distribute attention computation across devices, while sequence parallelism reduces memory footprint by splitting input sequences across workers. This design minimizes all-to-all communication overhead, enhances scalability for extremely long sequences, and enables efficient utilization of distributed hardware resources. Experimental analysis indicates that the proposed method achieves substantial throughput improvements compared to conventional data- and tensor-parallel approaches, particularly in scenarios with high sequence length and large model size.

## 1. Introduction and Problem Statement

Transformers face quadratic attention complexity and heavy memory requirements, making Multi-Head Attention (MHA) a bottleneck for distributed training and inference. This work proposes combining **Ring Attention** (ring topology replacing global communication) with **sequence parallelism** (splitting input sequences across devices) to create a balanced parallelization scheme for large-scale, memory-constrained environments.

## 2. Technical Methodology

### 2.1 Problem Setup
- **Input**: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ where $B$=batch, $L$=sequence length, $d_{\text{model}}$=hidden size
- **MHA**: $H$ heads, $d_h = d_{\text{model}}/H$ per head
- **Distributed**: $P$ devices $\{D_0, ..., D_{P-1}\}$

### 2.2 Sequence Parallelism
- Split sequence dimension: $X = [X^{(0)}, ..., X^{(P-1)}]$ with $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$
- **Memory reduction**: Activation memory reduced by factor $P$

### 2.3 Ring Attention Algorithm
**Ring topology** with $P$ stages:
1. **Initialize**: Each device computes $Q^{(p)}, K^{(p)}, V^{(p)}$ from local $X^{(p)}$
2. **Ring communication**: 
   - Stage $t$: Compute partial attention with current $K^{(\text{src})}, V^{(\text{src})}$
   - Pass $K,V$ to next device: $\text{src} = (p-t) \bmod P$
   - Accumulate results over $P$ stages
3. **Result**: Each device has full attention for its local queries

### 2.4 Communication Analysis
- **All-gather baseline**: $\mathcal{O}(L d_{\text{model}})$ per device
- **Ring Attention**: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per stage, $P$ stages total
- **Memory**: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ vs $\mathcal{O}(L d_{\text{model}})$

## 3. Experimental Results

### 3.1 Setup
- **Hardware**: 16×H100 GPUs with NVLink/NVSwitch
- **Model**: Dense Transformer (4 layers)
- **Fixed**: FP16, batch=1024 tokens, 16 heads×512 dim, MLP=32768
- **Baseline**: TP=8, PP=2 (no sequence/ring)
- **Proposed**: RA+SP (Ring Attention + Sequence Parallelism)

### 3.2 Performance
| Method | TPS (tokens/s) | TPOT (ms) |
|--------|----------------|-----------|
| Baseline | 1.20M | 0.85 |
| RA+SP | **1.45M** | **0.70** |

**Improvements**: 20.8% TPS increase, 17.6% TPOT reduction

## 4. Implementation Details
- **Communication**: NCCL `send/recv` or MPI point-to-point
- **Overlap**: Attention computation overlaps with async KV transfers
- **Precision**: Mixed fp16/bf16 for Q,K,V tensors
- **Scalability**: Benefits increase with $L>16k$ tokens and larger $P$