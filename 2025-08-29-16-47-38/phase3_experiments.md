# Phase 3: Experiments Extraction - Two-Level Attention Partitioning

## Experimental Setup

### Hardware Configuration
- **System**: 16 NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16) for throughput and numerical stability balance

### Model Configurations
Two transformer variants tested:
1. **2-layer Dense Transformer model**
2. **2-layer Mixture-of-Experts (MoE) Transformer model** with 4 experts per layer

### Fixed Parameters
- **Batch size**: 1024 (fixed across all tests)
- **Number of heads**: 16 (fixed)
- **Dimension per head**: 512 (fixed)
- **MLP hidden size**: 32768 (fixed)

## Baseline Configuration
- **Method**: Tensor Parallelism (TP) + Pipeline Parallelism (PP)
- **TP degree**: 8
- **PP degree**: 2
- **Total GPUs utilized**: 16 (8×2)
- **Description**: Widely adopted method for large-scale model deployment

## Evaluation Metrics
1. **Throughput (TPS)**: Tokens processed per second
2. **Time Per Output Token (TPOT)**: Average synchronization and communication overhead per token (milliseconds)

## Experimental Results

### Results Table
| Model Type | Method | TPS (tokens/sec) | TPOT (ms) |
|------------|--------|------------------|-----------|
| 4-layer Dense | Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| 4-layer Dense | Proposed (m×n=16) | 1,580,000 | 0.22 |
| 4-layer MoE | Baseline (TP=8, PP=2) | 850,000 | 0.45 |
| 4-layer MoE | Proposed (m×n=16) | 1,150,000 | 0.30 |

## Performance Analysis

### Dense Model Results
- **Throughput improvement**: +31.7% (1.2M → 1.58M tokens/sec)
- **Overhead reduction**: -37.1% (0.35ms → 0.22ms TPOT)

### MoE Model Results  
- **Throughput improvement**: +35.3% (850K → 1.15M tokens/sec)
- **Overhead reduction**: -33.3% (0.45ms → 0.30ms TPOT)

## Key Insights from Results

### Hardware Utilization
- Proposed method achieves **full utilization** of all 16 GPUs via m×n=16 partitions
- Baseline uses TP8+PP2 which creates different parallelization patterns
- Finer granularity enables better load balancing across devices

### Communication Efficiency
- Decreased TPOT (Time Per Output Token) reflects:
  - Reduced synchronization costs
  - More efficient communication patterns
  - Better localization of computations

### Model Type Impact
- Both Dense and MoE models show significant improvements
- MoE model shows slightly higher relative throughput gains (35.3% vs 31.7%)
- Consistent performance benefits across different model architectures

## Experimental Configuration Details

### Proposed Method Parameters
- **m×n = 16**: Implies m=4, n=4 configuration (4 dimension slices × 4 head groups)
- **Partition mapping**: Each of 16 partitions assigned to 1 GPU
- **Head group size**: h_g = h/n = 16/4 = 4 heads per group
- **Dimension slice size**: d_s = d/m = 512/4 = 128 dimensions per slice

### Baseline Configuration Details
- **Tensor Parallelism (TP=8)**: Splits model tensors across 8 GPUs
- **Pipeline Parallelism (PP=2)**: Splits model layers across 2 pipeline stages
- **GPU mapping**: Each pipeline stage uses 8 GPUs for tensor parallelism

## Discussion Points

### Throughput Saturation
- Large batch size (1024) ensures GPU saturation
- FP16 precision maximizes throughput without hardware idling
- Performance gains attributed to parallelization strategy improvements

### Scalability Implications
- Method scales to m×n devices regardless of original head count
- Enables deployment scenarios where devices ≫ heads
- Provides pathway for very large-scale distributed inference