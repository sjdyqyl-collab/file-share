# Ring Attention with Sequence Parallelism: A Concise Paper

## Abstract
We present a novel parallelization strategy for Multi-Head Attention (MHA) in large-scale transformer models that combines Ring Attention with sequence parallelism. Our approach leverages the communication-efficient properties of the ring topology to distribute attention computation across devices, while sequence parallelism reduces memory footprint by splitting input sequences across workers. This design minimizes all-to-all communication overhead, enhances scalability for extremely long sequences, and enables efficient utilization of distributed hardware resources. Experimental analysis indicates that the proposed method achieves substantial throughput improvements compared to conventional data- and tensor-parallel approaches, particularly in scenarios with high sequence length and large model size.

## 1. Problem Statement
Transformers face quadratic attention complexity and heavy memory requirements. Multi-Head Attention (MHA) becomes a bottleneck due to communication-intensive operations, especially when scaling to trillions of parameters or handling extremely long sequences (L > 16k tokens).

## 2. Proposed Solution
### 2.1 Ring Attention
- **Topology**: Devices arranged in logical ring
- **Communication**: P stages of peer-to-peer exchanges instead of all-gather
- **Benefits**: Lower peak bandwidth, better communication-computation overlap

### 2.2 Sequence Parallelism
- **Data Split**: Sequence dimension L divided across P devices
- **Memory Reduction**: Activation memory drops from O(L·d_model) to O(L/P·d_model)
- **Per Device Storage**: Each device stores X^(p) ∈ ℝ^(B×L/P×d_model)

## 3. Methodology

### 3.1 Notation and Setup
- Input: X ∈ ℝ^(B×L×d_model)
- Heads: H heads, each with dimension d_h = d_model/H
- Devices: P distributed devices {D_0, D_1, ..., D_{P-1}}

### 3.2 Combined Algorithm
```
for p in parallel on devices:
    Q_p, K_p, V_p = Project(X_p)  # Local projections
    output_p = 0
    KV_block = (K_p, V_p)
    
    for t in 0..P-1:
        src_idx = (p - t) mod P
        partial = Attention(Q_p, KV_block)
        output_p += partial
        send KV_block to next device
        receive KV_block from previous device
```

### 3.3 Communication Complexity
- **Naive All-Gather**: O(L·d_model) per device per step
- **Ring Attention**: O(L/P·d_model) per stage, P stages total
- **Memory**: 16× reduction with P=16 devices

### 3.4 Implementation Details
- **Backend**: NCCL send/recv primitives or MPI point-to-point
- **Precision**: Mixed-precision (fp16/bf16) for Q,K,V
- **Overlap**: Asynchronous communication with computation
- **Fused Kernels**: Projection + softmax + communication hooks

## 4. Experiments

### 4.1 Setup
- **Hardware**: 16× NVIDIA H100 GPUs with NVLink/NVSwitch
- **Model**: Dense Transformer (4 layers, 16 heads, 512 head dim, 32768 MLP hidden)
- **Precision**: FP16, Batch size: 1024 tokens
- **Baseline**: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)

### 4.2 Results
| Model | Method | TPS (tokens/s) | TPOT (ms) |
|-------|--------|----------------|-----------|
| Dense (4L) | Baseline | 1.20M | 0.85 |
| Dense (4L) | RA+SP | **1.45M** | **0.70** |

**Improvements**: 20.8% TPS increase, 17.6% TPOT reduction

## 5. Deployment Configuration Summary

### 5.1 Baseline Model
- **Parallel Strategy**: Tensor Parallelism (8) + Pipeline Parallelism (2)
- **Device Mapping**: 2 pipeline stages × 8 tensor parallel devices
- **Memory**: Full sequence stored on each tensor parallel group

### 5.2 RA+SP Model
- **Parallel Strategy**: Ring Sequence Parallelism (16)
- **Device Mapping**: 16 devices in ring topology
- **Memory**: Sequence split 1024 tokens per device (16384/16)
- **Communication**: 16-stage ring with KV blocks of size 1024×512×2 bytes

## 6. Key Technical Parameters
- **Sequence Length**: 16384 tokens
- **Hidden Size**: 8192 (16 heads × 512)
- **Ring Stages**: 16
- **Memory Reduction**: 16× via sequence parallelism
- **Communication**: NCCL send/recv with overlap
- **Optimal Range**: L > 16k tokens, P ≥ 16 devices