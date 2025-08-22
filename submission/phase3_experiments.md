# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Configuration
- **System**: 16 NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16) for throughput and numerical stability
- **Batch Size**: Fixed at 1024
- **Model Parameters**:
  - Number of heads: 16
  - Dimension per head: 512
  - MLP hidden size: 32768

### Model Architectures Tested
1. **4-layer Dense Transformer model**
2. **4-layer Mixture-of-Experts (MoE) Transformer model** (8 experts per layer)

### Baseline Configuration
- **Tensor Parallelism (TP)**: Degree 8
- **Pipeline Parallelism (PP)**: Degree 2
- **Total**: TP=8 + PP=2 = 16 GPUs (fully utilized)
- This represents a widely adopted method for large-scale model deployment

### Evaluation Metrics
- **Throughput (TPS)**: Tokens processed per second
- **Time Per Output Token (TPOT)**: Average synchronization and communication overhead time per token (milliseconds)

## Experimental Results

### Results Table
| Model Type | Method | TPS (tokens/sec) | TPOT (ms) |
|------------|--------|------------------|-----------|
| 4-layer Dense | Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| 4-layer Dense | Proposed (m×n=16) | 1,580,000 | 0.22 |
| 4-layer MoE | Baseline (TP=8, PP=2) | 850,000 | 0.45 |
| 4-layer MoE | Proposed (m×n=16) | 1,150,000 | 0.30 |

## Performance Analysis

### Dense Transformer Results
- **Throughput improvement**: 31.7% (1.2M → 1.58M tokens/sec)
- **Communication overhead reduction**: 37.1% (0.35ms → 0.22ms TPOT)

### MoE Transformer Results
- **Throughput improvement**: 35.3% (850K → 1.15M tokens/sec)
- **Communication overhead reduction**: 33.3% (0.45ms → 0.30ms TPOT)

## Key Findings

### Performance Benefits
- Consistent improvements across both dense and MoE architectures
- Higher throughput gains for MoE models (35.3% vs 31.7%)
- Significant reduction in synchronization and communication overhead

### Hardware Utilization
- Proposed method fully exploits all 16 GPUs through m×n=16 partitions
- Achieves better hardware utilization compared to TP=8+PP=2 baseline
- Decreased TPOT reflects reduced synchronization costs and more efficient communication patterns

### Experimental Conditions
- Large batch size (1024) and FP16 precision ensure GPU saturation
- Performance gains attributed to parallelization strategy improvements rather than hardware idling
- Results validate effectiveness of combining head-wise and intra-head dimension-wise slicing

## Discussion
- Finer granularity enables better load balancing
- Reduced cross-device communication through localized computations
- Method effectively leverages large-scale distributed infrastructure
- Results demonstrate practical applicability for both dense and sparse transformer models