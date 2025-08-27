# Helix: Two-Level Attention Partitioning for Large-Scale Transformer Models

### Abstract

We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Introduction

Transformer architectures face scaling challenges when distributing MHA across devices. Traditional head-wise partitioning is limited by fixed head count (h). We introduce a two-level partitioning strategy that splits h heads into n groups and further slices each head's d dimensions into m segments, enabling m×n partitions for m×n devices.

## Method

### Multi-Head Attention Recap
- Input: X ∈ ℝ^(B×L×D) where D = h×d
- MHA computes: Q,K,V = XW_Q, XW_K, XW_V with W ∈ ℝ^(D×D)
- Each head i: Attention_i = softmax(Q_i K_i^T/√d)V_i

### Two-Level Partitioning
1. **Head-level**: Split h heads into n groups → h_g = h/n heads per group
2. **Dimension-level**: Split d dimensions into m segments → d_s = d/m per segment
3. **Total partitions**: m×n, each handling h_g heads × d_s dimensions

### Implementation
- Weight matrices W_Q,W_K,W_V partitioned into blocks W^(i,j) ∈ ℝ^(d_s·h_g × d_s·h_g)
- Each device (i,j) computes: Attention^(i,j) = softmax(Q^(i,j)(K^(i,j))^T/√d_s)V^(i,j)
- Aggregation: Concatenate dimension slices within groups, then concatenate head groups

## Experiments

### Setup
- Hardware: 16 NVIDIA H100 GPUs, FP16 precision
- Models: 2-layer Dense Transformer, 2-layer MoE Transformer (4 experts/layer)
- Fixed: h=16 heads, d=512/head, batch=1024, hidden=32768

### Results
| Model | Method | TPS (tokens/sec) | TPOT (ms) |
|-------|--------|------------------|-----------|
| Dense | Baseline (TP=8,PP=2) | 1,200,000 | 0.35 |
| Dense | Proposed (m×n=16) | 1,580,000 | 0.22 |
| MoE   | Baseline (TP=8,PP=2) | 850,000 | 0.45 |
| MoE   | Proposed (m×n=16) | 1,150,000 | 0.30 |

### Key Findings
- Dense model: +31.7% throughput, -37.1% overhead
- MoE model: +35.3% throughput, -33.3% overhead
- Method scales beyond head count limitations

## Conclusion

Our two-level partitioning enables efficient deployment of MHA across m×n devices by combining head-wise and dimension-wise splitting. Experiments demonstrate significant throughput improvements (31-35%) and reduced communication overhead (33-37%) compared to TP+PP baselines, validating the approach for large-scale transformer deployment.