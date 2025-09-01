# Experiments: Two-Level Attention Partitioning Evaluation

## Experimental Setup

### Hardware Configuration
- **System**: 16 NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16) for throughput and numerical stability balance

### Model Configurations
Two model types tested:
1. **2-layer Dense Transformer model**
2. **2-layer Mixture-of-Experts (MoE) Transformer model** with 4 experts per layer

### Fixed Parameters
- **Batch size**: 1024
- **Number of heads**: 16
- **Dimension per head**: 512
- **Hidden size of MLP**: 32768

## Baseline Configuration

### Baseline Method
- **Tensor Parallelism (TP)**: degree 8
- **Pipeline Parallelism (PP)**: degree 2
- **Total GPUs**: 16 (TP=8 + PP=2)
- **Description**: Widely adopted method for large-scale model deployment

## Evaluation Metrics

### Primary Metrics
1. **Throughput (TPS)**: Tokens processed per second
2. **Time Per Output Token (TPOT)**: Average synchronization and communication overhead time per token (milliseconds)

## Results

### Performance Comparison Table

| Model Type | Method | TPS (tokens/sec) | TPOT (ms) |
|------------|--------|------------------|-----------|
| 4-layer Dense | Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| 4-layer Dense | Proposed (m×n=16) | 1,580,000 | 0.22 |
| 4-layer MoE | Baseline (TP=8, PP=2) | 850,000 | 0.45 |
| 4-layer MoE | Proposed (m×n=16) | 1,150,000 | 0.30 |

### Detailed Analysis

#### Dense Model Results
- **Throughput improvement**: 31.7% increase (1.2M → 1.58M tokens/sec)
- **Overhead reduction**: 37.1% decrease (TPOT from 0.35ms → 0.22ms)

#### MoE Model Results
- **Throughput improvement**: 35.3% increase (850K → 1.15M tokens/sec)
- **Overhead reduction**: 33.3% decrease (TPOT from 0.45ms → 0.30ms)

## Technical Analysis

### Partitioning Configuration
- **Proposed method**: m×n = 16 partitions
- **Mapping**: Each partition mapped to 1 GPU (total 16 GPUs utilized)
- **Granularity**: Fine-grained partitioning at both head and dimension levels

### Performance Factors
1. **Hardware utilization**: Full exploitation of 16 GPUs through m×n=16 mapping
2. **Load balancing**: Even distribution across head groups and dimension slices
3. **Communication efficiency**: Reduced synchronization cost and efficient communication patterns
4. **Memory efficiency**: Each device handles 1/16 of total computation

### Comparison with Baseline
- **Traditional TP+PP**: Coarser granularity with potential idle resources
- **Proposed method**: Finer granularity enabling better resource utilization
- **Communication**: Hierarchical partitioning reduces cross-device synchronization

## Experimental Validations

### Precision and Saturation
- **FP16 precision**: Maintains numerical stability while maximizing throughput
- **Large batch size (1024)**: Ensures GPU saturation, validating gains from parallelization strategy
- **Hardware idling**: Minimized through optimal partition mapping

### Scalability Demonstration
- **Linear scaling**: m×n=16 partitions demonstrate linear scalability to 16 devices
- **Beyond head limits**: Method works effectively even when m×n > h (16 > 16 heads)
- **Flexibility**: Can adapt to different cluster sizes by adjusting m and n

## Reproducibility Details

### Model Specifications
- **Dense model**: 4 layers (2 transformer layers as described)
- **MoE model**: 4 layers with 4 experts per layer
- **Attention mechanism**: Standard multi-head attention with 16 heads
- **Embedding dimensions**: 8192 total (16 heads × 512 dimensions/head)

### Environment Setup
- **GPU topology**: 16× H100 GPUs with high-bandwidth interconnect
- **Framework**: Compatible with existing model parallel frameworks
- **Communication**: Custom tensor partitioning and communication primitives
- **Memory**: Sufficient GPU memory for FP16 precision with batch size 1024

## Key Findings

### Quantitative Improvements
- **Average throughput gain**: ~33.5% across both model types
- **Average overhead reduction**: ~35.2% in communication time
- **Consistent performance**: Benefits observed across both dense and MoE architectures

### Qualitative Benefits
- **Scalability**: Method scales beyond traditional head-wise limitations
- **Flexibility**: Adapts to varying hardware configurations
- **Efficiency**: Better resource utilization through fine-grained partitioning
- **Practicality**: Compatible with existing distributed training frameworks