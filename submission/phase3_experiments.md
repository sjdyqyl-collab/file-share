# Phase 3: Experiments Extraction - Helix: Two-Level Attention Partitioning

## Experimental Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16) for throughput and numerical stability
- **Models tested**:
  - 2-layer Dense Transformer model
  - 2-layer Mixture-of-Experts (MoE) Transformer model with 4 experts per layer
- **Fixed parameters**:
  - Batch size: 1024
  - Number of heads: 16
  - Head dimension: 512
  - MLP hidden size: 32768

## Baseline Configuration
- **Method**: Tensor Parallelism (TP) with degree 8 + Pipeline Parallelism (PP) with degree 2
- **Total GPUs**: 16 (TP=8 × PP=2)
- **Description**: Widely adopted method for large-scale model deployment

## Metrics
- **Throughput (TPS)**: Tokens processed per second
- **Time Per Output Token (TPOT)**: Average synchronization and communication overhead time per token (milliseconds)

## Results Table

| Model Type    | Method                | TPS (tokens/sec) | TPOT (ms) |
| ------------- | --------------------- | ---------------- | --------------- |
| 4-layer Dense | Baseline (TP=8, PP=2) | 1,200,000        | 0.35            |
| 4-layer Dense | Proposed (m×n=16)     | 1,580,000        | 0.22            |
| 4-layer MoE   | Baseline (TP=8, PP=2) | 850,000          | 0.45            |
| 4-layer MoE   | Proposed (m×n=16)     | 1,150,000        | 0.30            |

## Performance Analysis

### Dense Model Results
- **Throughput improvement**: 31.7% increase (1.2M → 1.58M tokens/sec)
- **Overhead reduction**: 37.1% decrease in TPOT (0.35ms → 0.22ms)

### MoE Model Results
- **Throughput improvement**: 35.3% increase (850K → 1.15M tokens/sec)
- **Overhead reduction**: 33.3% decrease in TPOT (0.45ms → 0.30ms)

## Discussion
- **Hardware utilization**: Two-level partitioning fully exploits 16 GPUs by mapping m×n=16 partitions to devices
- **Synchronization cost**: Decreased TPOT reflects reduced synchronization cost and more efficient communication patterns
- **Throughput saturation**: FP16 precision and large batch size (1024) ensure performance gains come from parallelization strategy improvements rather than hardware idling
- **Fine-grained partitioning**: Enables better load balancing and reduces cross-device communication compared to baseline TP=8 + PP=2 scheme

## Key Findings
- Proposed method achieves substantial improvements in inference throughput (up to 35%)
- Communication overhead reduced by over 30% compared to baseline
- Method validates effectiveness of combining head-wise and intra-head dimension-wise slicing
- Better workload balancing and hardware resource utilization demonstrated