# FA Pool: A Dynamic Parallel Strategy for Scaling Attention Mechanisms in Large Language Models

## Abstract

The computational complexity of attention mechanisms in transformer-based models grows quadratically with sequence length, creating a significant bottleneck for processing long sequences. We propose FA Pool (Flash Attention Pool), a novel dynamic parallel strategy that intelligently allocates GPU resources based on sequence length thresholds. When input sequences exceed a predetermined length, FA Pool activates additional GPU resources to form a computation pool dedicated to parallel attention calculations, thereby reducing the computational burden on individual GPUs. Our approach combines the benefits of Flash Attention's memory-efficient algorithms with dynamic resource allocation to achieve superior scaling characteristics. Experimental results on a 4-layer Dense model demonstrate that FA Pool achieves significant improvements in both Time Per Output Token (TPOT) and Tokens Per Second (TPS) metrics compared to traditional static parallelization strategies (TP=8, PP=2 baseline). The strategy shows particular effectiveness for long sequence processing, achieving up to 3.2x improvement in TPOT and 2.8x improvement in TPS for sequences exceeding 8K tokens.

## 1. Introduction

Large language models face a fundamental challenge: attention mechanisms exhibit O(n²) complexity with sequence length, creating computational bottlenecks for long sequences. Traditional static parallelization strategies (TP=8, PP=2) lead to suboptimal resource utilization due to the mismatch between fixed resource allocation and dynamic computational requirements.

## 2. FA Pool Methodology

### 2.1 System Architecture

**Base Layer (8 GPUs - Fixed)**
- Embedding layer, positional encoding, output layers
- Feed-forward networks (linear complexity, no parallelization needed)
- Maintains model coherence

**Attention Pool (Up to 32 GPUs - Dynamic)**
- Dedicated to attention computation
- Activated when sequence length > 4096 tokens
- Released when sequence length drops below threshold

**Resource Manager**
- Monitors sequence length continuously
- Manages GPU allocation/deallocation
- Coordinates workload distribution

### 2.2 Dynamic Resource Allocation Strategy

**Threshold Determination**
```
Threshold = argmin_t (Attention_Time(t) > FFN_Time + Overhead)
```
Empirically determined: 4096 tokens

**Allocation Mechanism**
1. Monitor sequence length during inference
2. Compare against 4096 token threshold
3. Activate additional GPUs when exceeded
4. Partition attention computation across pool
5. Aggregate results via concatenation
6. Release resources when below threshold

### 2.3 Attention Parallelization

**Block-wise Parallelization**
```
Input: Q, K, V, sequence length n, pool GPUs p
Process:
1. Block size: b = ceil(n / p)
2. Each GPU i computes: O_i = FlashAttention(Q_i, K, V)
3. Aggregate: O = concat(O_0, O_1, ..., O_p-1)
```

**Communication Optimization**
- KV cache sharing across pool GPUs
- Asynchronous execution overlapping with FFN
- Hierarchical reduction (tree-based, O(log p) steps)

### 2.4 Model Configuration

**Architecture**: 4-layer Dense model
- **Parameters**: ~13B
- **Hidden Dimension**: 4096
- **Attention Heads**: 32
- **Feed-forward Dimension**: 16384
- **Activation**: GELU
- **Normalization**: Pre-norm RMSNorm

## 3. Experimental Results

### 3.1 Setup

**Baseline**: TP=8, PP=2 (16 GPUs total)
**FA Pool**: 8 base GPUs + up to 32 pool GPUs
**Hardware**: NVIDIA A100 80GB, NVLink 3.0, InfiniBand

### 3.2 Performance Results

**TPOT Improvements**
| Length | Baseline | FA Pool | Speedup |
|--------|----------|---------|---------|
| 512    | 45ms     | 41ms    | 1.1x    |
| 2048   | 78ms     | 56ms    | 1.4x    |
| 8192   | 245ms    | 117ms   | 2.1x    |
| 16384  | 892ms    | 279ms   | 3.2x    |

**TPS Improvements**
| Length | Baseline | FA Pool | Speedup |
|--------|----------|---------|---------|
| 512    | 22.2     | 26.7    | 1.2x    |
| 2048   | 25.6     | 41.0    | 1.6x    |
| 8192   | 33.4     | 83.5    | 2.5x    |
| 16384  | 18.3     | 51.2    | 2.8x    |

### 3.3 Resource Utilization
- **GPU Utilization**: 85-92% (FA Pool) vs 45-60% (baseline)
- **Memory Usage**: 65GB (base), 45GB (pool GPUs)
- **Communication Overhead**: <15% of total time
- **Optimal Pool Size**: 24 GPUs (performance plateau)

### 3.4 Scaling Characteristics
- **Strong Scaling**: Near-linear up to 16K tokens
- **Resource Efficiency**: 40-47% improvement over baseline
- **Memory Distribution**: Balanced across base and pool GPUs

## 4. Conclusion

FA Pool demonstrates significant improvements for long sequence processing through dynamic resource allocation. The strategy achieves up to 3.2x TPOT and 2.8x TPS improvements for sequences exceeding 8K tokens, with 85-92% GPU utilization compared to 45-60% for static strategies. The approach is particularly effective for variable sequence workloads in real-world deployment scenarios.