# Helix: Two-Level Attention Partitioning - Experiments

## Experimental Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16)
- **Model**: 2-layer Dense Transformer
- **Batch size**: 1024
- **Heads**: 16
- **Head dimension**: 512
- **MLP hidden size**: 32768
- **Task**: Inference

## Baseline Configuration
- **Method**: Tensor Parallelism (TP) + Pipeline Parallelism (PP)
- **TP degree**: 8
- **PP degree**: 2
- **Total devices**: 16 (8 × 2)
- **Description**: Widely adopted method for large-scale model deployment

## Proposed Configuration
- **Method**: Two-level partitioning
- **Partitioning**: m × n = 16 partitions
- **Mapping**: 16 partitions to 16 devices (1:1 mapping)
- **Head partitioning (n)**: 4 groups (h_g = 4 heads per group)
- **Dimension partitioning (m)**: 4 slices (d_s = 128 per slice)

## Metrics
- **TPS (Tokens Per Second)**: Throughput measurement
- **TPOT (Time Per Output Token)**: Average synchronization and communication overhead per token (milliseconds)

## Results

### Dense Transformer Results
| Method | TPS (tokens/sec) | TPOT (ms) |
|--------|------------------|-----------|
| Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| Proposed (m×n=16) | 1,580,000 | 0.22 |

### Performance Improvements
- **Throughput improvement**: 31.7% increase (1.2M → 1.58M tokens/sec)
- **Overhead reduction**: 37.1% decrease (0.35ms → 0.22ms TPOT)

## Analysis
- **Hardware utilization**: Proposed method fully exploits all 16 GPUs through 1:1 partition-to-device mapping
- **Load balancing**: Finer granularity (m×n=16) vs baseline (TP=8, PP=2) enables better workload distribution
- **Communication efficiency**: Reduced synchronization cost evidenced by lower TPOT
- **Memory efficiency**: Each device stores only fraction of MHA parameters and activations

## Discussion
- **FP16 precision**: Maintains numerical stability while maximizing throughput
- **Large batch size (1024)**: Ensures GPU saturation, isolating performance gains to parallelization strategy
- **Scalability**: Method scales beyond traditional head-wise splitting limits
- **Deployment flexibility**: m and n can be adjusted based on hardware topology and network bandwidth