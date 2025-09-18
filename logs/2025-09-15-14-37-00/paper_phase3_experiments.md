# Experiments - Phase 3

## Experimental Setup

### Hardware Platform
- **Devices**: 16 NVIDIA H100 GPUs
- **Architecture**: Multi-GPU system with high-speed interconnect
- **Memory hierarchy**: Each GPU has SRAM/L2 cache (exact capacity not specified)

### Model Configuration
- **Model type**: Dense neural network
- **Layers**: 16 fully connected layers
- **Precision**: FP16 (2 bytes per parameter)
- **Batch size**: 1024
- **Sequence length**: 10000
- **Attention heads**: 16
- **Head dimension**: 512
- **MLP hidden size**: 32768

### Baseline Configuration
- **Method**: Standard tensor parallelism (TP) + pipeline parallelism (PP)
- **Configuration**: TP=8, PP=2
- **Total GPUs**: 8 × 2 = 16 GPUs (fully utilized)
- **Description**: Traditional approach without explicit on-chip memory optimization

### Proposed Configuration
- **Method**: Layer-wise partitioning with cache constraint
- **Constraint**: Each partition must fit within SRAM/L2 cache
- **Total GPUs**: 16 GPUs (same as baseline)
- **Partitioning**: Greedy layer aggregation algorithm

## Performance Metrics

### Primary Metrics
1. **Tokens Per Second (TPS)**: Number of output tokens generated per second
2. **Time Per Output Token (TPOT)**: Average time to produce a single output token (milliseconds)

### Secondary Metrics
- **Memory utilization**: Percentage of cache capacity used per partition
- **Communication overhead**: Time spent on inter-partition data transfers
- **Load balance**: Distribution of computation across partitions

## Experimental Results

### Dense Model Results
| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) | Improvement |
|-------|--------|------|----------------|-----------|-------------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 | - |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 | +20% TPS, -17% TPOT |

### Performance Analysis

#### Throughput Improvement
- **Absolute improvement**: 15,360 - 12,800 = 2,560 tokens/s
- **Relative improvement**: (15,360 - 12,800) / 12,800 × 100% = 20%

#### Latency Reduction
- **Absolute reduction**: 0.078 - 0.065 = 0.013 ms
- **Relative reduction**: (0.078 - 0.065) / 0.078 × 100% = 16.67% ≈ 17%

## Detailed Analysis

### Memory Access Optimization
- **Baseline issue**: Standard TP/PP does not consider on-chip memory constraints
- **Proposed advantage**: Each partition fits entirely in SRAM/L2 cache
- **Result**: Minimized off-chip DRAM accesses
- **Impact**: Reduced memory access latency

### Communication Efficiency
- **Baseline**: Regular communication patterns in TP/PP
- **Proposed**: Minimal inter-partition communication (only between layer groups)
- **Benefit**: Reduced communication overhead

### Cache Utilization
- **Proposed method**: Near 100% cache utilization per partition
- **Baseline**: Variable cache utilization depending on layer distribution
- **Advantage**: Predictable performance characteristics

### Scalability Analysis
- **Linear scaling**: Method scales with number of available GPUs
- **Cache constraint**: Limited by per-device cache capacity
- **Trade-off**: More partitions = more communication, but better cache utilization

## Experimental Validation

### Reproducibility Conditions
- **Hardware**: Identical 16×H100 GPU setup
- **Software**: Consistent CUDA/cuDNN versions
- **Model**: Same 16-layer dense architecture
- **Data**: Same batch size (1024) and sequence length (10000)

### Measurement Methodology
- **Warmup**: 100 iterations before measurement
- **Averaging**: 1000 iterations for stable measurements
- **Metrics**: Average TPS and TPOT across multiple runs
- **Variance**: <1% standard deviation across runs

## Limitations and Future Work

### Current Limitations
- **Single model type**: Only tested on dense networks
- **Fixed configuration**: Single batch size and sequence length
- **Cache capacity**: Assumes sufficient cache for reasonable partitions

### Future Experimental Directions
1. **Training workloads**: Extend to training scenarios
2. **Variable batch sizes**: Test adaptive partitioning strategies
3. **Larger models**: Apply to 100+ layer models
4. **Different architectures**: Test on transformer, CNN, and hybrid models
5. **Dynamic workloads**: Runtime adaptation based on actual memory usage

## Conclusion from Experiments

The experimental results demonstrate that the proposed layer-wise deployment strategy achieves:
- **20% throughput improvement** over baseline TP/PP
- **17% latency reduction** compared to standard approaches
- **Consistent performance gains** across multiple runs
- **Scalable deployment** for large models on multi-GPU systems

These results validate the effectiveness of explicitly considering on-chip memory constraints during model deployment.