# Phase Three: Experiments Extraction

## Experimental Setup

### Hardware Configuration
- **System**: 16 NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16) for throughput and numerical stability balance

### Model Architectures Tested
1. **2-layer Dense Transformer model**
2. **2-layer Mixture-of-Experts (MoE) Transformer model** with 4 experts per layer

### Fixed Parameters (All Tests)
- **Batch size**: 1024
- **Number of heads**: 16
- **Dimension per head**: 512
- **MLP hidden size**: 32768

## Baseline Configuration
- **Method**: Tensor Parallelism (TP) + Pipeline Parallelism (PP)
- **TP degree**: 8
- **PP degree**: 2
- **Total GPUs utilized**: 16 (8×2)
- **Description**: Widely adopted method for large-scale model deployment

## Evaluation Metrics
1. **Throughput (TPS)**: Tokens processed per second
2. **Time Per Output Token (TPOT)**: Average synchronization and communication overhead time per token (milliseconds)

## Experimental Results

### Dense Transformer Model
| Method | TPS (tokens/sec) | TPOT (ms) |
|--------|------------------|-----------|
| Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| Proposed (m×n=16) | 1,580,000 | 0.22 |

**Improvements**:
- **Throughput**: +31.7% (1.2M → 1.58M tokens/sec)
- **Overhead Reduction**: -37.1% (0.35ms → 0.22ms)

### MoE Transformer Model
| Method | TPS (tokens/sec) | TPOT (ms) |
|--------|------------------|-----------|
| Baseline (TP=8, PP=2) | 850,000 | 0.45 |
| Proposed (m×n=16) | 1,150,000 | 0.30 |

**Improvements**:
- **Throughput**: +35.3% (850K → 1.15M tokens/sec)
- **Overhead Reduction**: -33.3% (0.45ms → 0.30ms)

## Analysis Summary

### Performance Gains
- **Dense model**: 31.7% throughput improvement, 37.1% overhead reduction
- **MoE model**: 35.3% throughput improvement, 33.3% overhead reduction

### Key Factors for Improvement
1. **Fine-grained partitioning** enables better load balancing
2. **Reduced communication** compared to TP+PP baseline
3. **Better hardware utilization** with m×n=16 partitions mapping directly to 16 GPUs
4. **Decreased TPOT** reflects reduced synchronization cost and efficient communication patterns

### Experimental Controls
- **Precision**: FP16 maintained across all tests to ensure fair comparison
- **Batch size**: Large batch (1024) ensures GPU saturation
- **Performance gains** attributed to parallelization strategy improvements, not hardware idling

## Partition Configuration Inference
Based on m×n=16 partitions and 16 GPUs:
- **Likely configuration**: m=4, n=4
- **Head groups**: n=4 groups with h_g=16/4=4 heads per group
- **Dimension slices**: m=4 slices with d_s=512/4=128 dimensions per slice