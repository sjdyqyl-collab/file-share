# Phase 3: Detailed Experiments

## Experimental Setup

### Hardware Configuration
- **Platform**: 16 NVIDIA H100 GPUs
- **Memory Hierarchy**: Each GPU has SRAM/L2 cache (capacity C)
- **Interconnect**: High-speed links between GPUs for partition communication

### Model Specifications
- **Model Type**: Dense 16-layer fully connected network
- **Precision**: FP16 (2 bytes per parameter)
- **Batch Size**: 1024
- **Architecture Details**:
  - Number of layers: 16
  - Number of attention heads: 16
  - Dimension per head: 512
  - Hidden size: 8192 (16 × 512)
  - MLP hidden size: 32768

### Baseline Configuration
- **Method**: Standard tensor parallelism + pipeline parallelism
- **Configuration**: TP=8, PP=2
- **GPU Utilization**: 8 × 2 = 16 GPUs (full utilization)
- **Memory Distribution**: Model parameters split via tensor parallelism across 8 GPUs, pipeline stages across 2 GPUs

### Proposed Configuration
- **Method**: Layer-wise partitioning based on cache capacity
- **Configuration**: Variable k partitions based on greedy algorithm
- **Constraint**: Each partition must fit in SRAM/L2 cache capacity C
- **Partitioning**: 16 layers divided into k groups where S(Pᵢ) ≤ C

## Performance Metrics

### Primary Metrics
1. **Tokens Per Second (TPS)**: Output tokens generated per second
2. **Time Per Output Token (TPOT)**: Average time per token in milliseconds

### Results Table
| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

## Performance Analysis

### Quantitative Improvements
- **TPS Improvement**: (15,360 - 12,800) / 12,800 = 20% increase
- **TPOT Reduction**: (0.078 - 0.065) / 0.078 = 16.7% reduction
- **Latency Improvement**: 17% faster per token generation

### Root Cause Analysis
- **Baseline Issues**: 
  - Does not consider on-chip memory constraints
  - More off-chip DRAM accesses required
  - Higher inter-GPU communication overhead
- **Proposed Advantages**:
  - Entire partitions fit in fast SRAM/L2 cache
  - Minimal off-chip memory accesses
  - Reduced communication between partitions
  - Better memory locality and cache utilization

### Memory Footprint Estimation
For the 16-layer dense model:
- **Per-layer weight size**: Calculated based on layer dimensions
- **Activation size**: 1024 × hidden_size × 2 bytes
- **Buffer size**: Operator-dependent workspace
- **Total model size**: Sum of all 16 layers
- **Partition sizes**: Determined by greedy algorithm to fit within cache capacity C

## Experimental Validation
- **Reproducibility**: Fixed random seeds and deterministic execution
- **Measurement**: 1000 iterations averaged for stable metrics
- **Warmup**: 100 iterations excluded from measurements
- **Validation**: Results consistent across multiple runs

## Scalability Implications
- **Model Size**: Method scales to larger models by increasing k (partitions)
- **Hardware**: Scales with available GPUs (tested on 16, extensible to more)
- **Cache Size**: Benefits increase with larger cache capacity C
- **Batch Size**: Trade-off between batch size and partition count k