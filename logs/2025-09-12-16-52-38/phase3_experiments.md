# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Configuration
- System: 16 NVIDIA H100 GPUs
- Precision: Mixed precision (FP16)
- Framework: Compatible with existing model parallel frameworks

### Model Configuration
- **Model Type**: 2-layer Dense Transformer model
- **Batch Size**: 1024 (fixed)
- **Attention Heads**: 16 (fixed)
- **Head Dimension**: 512 (fixed)
- **Total Embedding Dimension**: 8192 (16 × 512)
- **MLP Hidden Size**: 32768 (fixed)

## Baseline Configuration
- **Method**: Tensor Parallelism (TP) + Pipeline Parallelism (PP)
- **TP Degree**: 8
- **PP Degree**: 2
- **Total GPUs**: 16 (8 × 2)
- **Description**: Widely adopted method for large-scale model deployment

## Proposed Method Configuration
- **Method**: Two-level attention partitioning
- **Partitioning**: m×n = 16 partitions
- **Mapping**: 1 partition per GPU (16 total GPUs)
- **Head Groups**: n = 4 (4 groups of 4 heads each)
- **Dimension Slices**: m = 4 (4 slices of 128 dimensions each)
- **Partition Size**: Each handles 4 heads × 128 dimensions = 512 dimensions

## Metrics
- **Throughput (TPS)**: Tokens processed per second
- **Time Per Output Token (TPOT)**: Average synchronization and communication overhead per token (milliseconds)

## Results

| Model Type | Method | TPS (tokens/sec) | TPOT (ms) |
|------------|--------|------------------|-----------|
| 2-layer Dense | Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| 2-layer Dense | Proposed (m×n=16) | 1,580,000 | 0.22 |

## Performance Analysis

### Throughput Improvement
- **Absolute Increase**: 1,580,000 - 1,200,000 = 380,000 tokens/sec
- **Relative Improvement**: (380,000/1,200,000) × 100 = **31.7%**

### Communication Overhead Reduction
- **Absolute Reduction**: 0.35 - 0.22 = 0.13 ms
- **Relative Reduction**: (0.13/0.35) × 100 = **37.1%**

## Key Findings
1. **Hardware Utilization**: Proposed method fully exploits all 16 GPUs through m×n=16 partitions
2. **Load Balancing**: Finer granularity enables better distribution of work
3. **Communication Efficiency**: Reduced synchronization costs due to localized partitions
4. **Scalability**: Method scales beyond head count limitations (16 vs 16 heads)
5. **Practical Viability**: Achieves significant improvements over strong baseline

## Discussion Points
- Large batch size (1024) and FP16 precision ensure GPU saturation
- Performance gains attributed to parallelization strategy, not hardware idling
- Method enables flexible scaling for very large clusters
- Future work includes extending to training scenarios and adaptive partitioning