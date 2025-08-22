# Phase 3: Experiments Extraction - Helix: Two-Level Attention Partitioning for Distributed Transformer Models

## Abstract (Retained as-is)
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Experimental Setup

### Hardware Configuration
- **System**: 16 NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16) for throughput and numerical stability balance

### Model Architectures Tested
1. **4-layer Dense Transformer model**
2. **4-layer Mixture-of-Experts (MoE) Transformer model**
   - 8 experts per layer

### Fixed Parameters Across All Tests
- **Batch size**: 1024
- **Number of heads**: 16
- **Dimension per head**: 512
- **Hidden size of MLP**: 32768

### Baseline Configuration
- **Tensor Parallelism (TP)**: Degree 8
- **Pipeline Parallelism (PP)**: Degree 2
- **Total GPUs utilized**: TP=8 × PP=2 = 16 GPUs
- **Description**: Widely adopted method for large-scale model deployment

### Proposed Method Configuration
- **Partitioning**: m×n = 16 partitions
- **Mapping**: Each partition assigned to one GPU (total 16 GPUs)
- **Configuration**: Two-level partitioning with head groups and dimension slices

## Evaluation Metrics

### Primary Metrics
1. **Throughput (TPS)**: Tokens processed per second
   - Higher is better
   - Measures overall system performance

2. **Time Per Output Token (TPOT)**: Average synchronization and communication overhead time per token
   - Unit: milliseconds (ms)
   - Lower is better
   - Reflects communication efficiency and synchronization costs

## Experimental Results

### Results Table
| Model Type   | Method                | TPS (tokens/sec) | TPOT (ms) |
|--------------|-----------------------|------------------|-----------|
| 4-layer Dense| Baseline (TP=8, PP=2) | 1,200,000        | 0.35      |
| 4-layer Dense| Proposed (m×n=16)     | 1,580,000        | 0.22      |
| 4-layer MoE  | Baseline (TP=8, PP=2) | 850,000          | 0.45      |
| 4-layer MoE  | Proposed (m×n=16)     | 1,150,000        | 0.30      |

## Performance Analysis

### Dense Transformer Results
- **Throughput improvement**: 31.7% increase
  - From 1.2M to 1.58M tokens/sec
- **Overhead reduction**: 37.1% decrease
  - From 0.35ms to 0.22ms TPOT

### MoE Transformer Results
- **Throughput improvement**: 35.3% increase
  - From 850K to 1.15M tokens/sec
- **Overhead reduction**: 33.3% decrease
  - From 0.45ms to 0.30ms TPOT

### Key Observations
- **Consistent improvements** across both dense and MoE architectures
- **Higher gains** for MoE model (35.3% vs 31.7%)
- **Significant communication overhead reduction** in both cases
- **Full hardware utilization** achieved with m×n=16 configuration

## Experimental Discussion

### Hardware Utilization
- Proposed method fully exploits all 16 GPUs
- Each GPU handles exactly one partition (m×n=16)
- Eliminates potential GPU idling present in baseline

### Communication Efficiency
- Decreased TPOT reflects reduced synchronization costs
- More efficient communication patterns compared to TP+PP baseline
- Hierarchical partitioning reduces cross-device bandwidth requirements

### Performance Factors
- **FP16 precision** ensures GPU saturation without numerical instability
- **Large batch size (1024)** helps maximize throughput
- **Performance gains** attributed to parallelization strategy improvements, not hardware idling

### Scalability Implications
- Results demonstrate effectiveness of two-level partitioning
- Method scales efficiently to 16 devices
- Approach can potentially scale to larger clusters with appropriate m,n selection

## Reproducibility Guidelines

### Required Parameters
- Model: 4-layer transformer (dense or MoE)
- Heads: 16
- Head dimension: 512
- MLP hidden size: 32768
- Batch size: 1024
- Precision: FP16
- GPUs: 16×H100

### Baseline Setup
- Tensor Parallelism degree: 8
- Pipeline Parallelism degree: 2
- Ensure proper placement to avoid pipeline bubbles

### Proposed Method Setup
- Configure m×n = 16 partitions
- Ensure equal distribution: m×n = 16
- Map each partition to individual GPU
- Implement hierarchical concatenation for result aggregation

### Measurement Protocol
- Warmup runs: ≥10 iterations
- Measurement runs: ≥100 iterations
- Report: Mean TPS and TPOT with standard deviation
- Monitor: GPU utilization and memory usage for validation