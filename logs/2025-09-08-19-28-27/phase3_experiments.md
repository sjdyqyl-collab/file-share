# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Platform
- **Platform**: 16 NVIDIA H100 GPUs
- **Memory**: Each GPU has SRAM/L2 cache capacity C (exact value not specified in paper)

### Model Configuration
- **Model Type**: Dense 16-layer fully connected network
- **Precision**: FP16 (2 bytes per parameter)
- **Batch Size**: 1024
- **Architecture Details**:
  - Number of attention heads: 16
  - Dimension per head: 512
  - Hidden size of MLP: 32768
  - Total layers: 16

### Baseline Comparison
- **Method**: Standard tensor parallelism (TP) + pipeline parallelism (PP)
- **Configuration**: TP=8, PP=2
- **Utilization**: 8 × 2 = 16 GPUs (fully utilized)

### Performance Metrics
1. **Tokens Per Second (TPS)**: Number of output tokens generated per second
2. **Time Per Output Token (TPOT)**: Average time to produce a single output token (milliseconds)

## Experimental Results

| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

## Performance Analysis

### Improvement Metrics
- **TPS Improvement**: 15,360 - 12,800 = 2,560 tokens/s (20% increase)
- **TPOT Reduction**: 0.078 - 0.065 = 0.013ms (17% reduction)
- **Relative Improvement**: 20% increase in throughput, 17% reduction in latency

### Root Cause Analysis
The performance improvement results from:
1. **Efficient on-chip memory utilization**: Each partition fits entirely within SRAM/L2 cache
2. **Reduced memory access latency**: Minimized off-chip DRAM accesses
3. **Lower communication overhead**: Less inter-card communication compared to TP+PP baseline
4. **Cache locality**: Better spatial and temporal locality of data access

### Baseline Limitations
The TP=8, PP=2 approach:
- Does not explicitly consider on-chip memory constraints
- Results in more off-chip memory accesses
- Incurs additional communication delays between tensor-parallel groups
- May have suboptimal memory locality compared to layer-wise partitioning

## Experimental Validation
The results demonstrate that the proposed layer-wise deployment strategy achieves substantial performance gains over traditional tensor and pipeline parallelism approaches, validating the effectiveness of cache-aware partitioning for large model deployment.