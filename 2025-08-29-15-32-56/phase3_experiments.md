# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Platform
- **GPUs**: 16 NVIDIA H100 GPUs
- **Memory**: SRAM/L2 cache per GPU (capacity C)

### Model Configurations
- **Dense Model**: 16-layer fully connected dense network
- **MoE Model**: 16-layer mixture-of-experts (MoE) model with 8 experts per layer
- **Precision**: FP16 (2 bytes per parameter)
- **Batch Size**: 1024
- **Heads**: 16
- **Head Dimension**: 512
- **MLP Hidden Size**: 32768

### Baseline Comparison
- **Method**: Standard tensor parallelism (TP) and pipeline parallelism (PP)
- **Configuration**: TP=8, PP=2
- **GPU Utilization**: 8 × 2 = 16 GPUs (full utilization)

### Performance Metrics
- **Tokens Per Second (TPS)**: Number of output tokens generated per second
- **Time Per Output Token (TPOT)**: Average time to produce a single output token (milliseconds)

## Results

| Model                    | Method                | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------------------------|-----------------------|------|----------------|-----------|
| Dense (16-layer)        | Baseline (TP=8, PP=2) | 16   | 12,800         | 0.078     |
| Dense (16-layer)        | Proposed Layer-wise   | 16   | 15,360         | 0.065     |
| MoE (16-layer, 8 experts) | Baseline (TP=8, PP=2) | 16   | 10,200         | 0.098     |
| MoE (16-layer, 8 experts) | Proposed Layer-wise   | 16   | 13,400         | 0.075     |

## Performance Analysis

### Dense Model Results
- **TPS Improvement**: 20% increase (12,800 → 15,360)
- **TPOT Reduction**: 17% reduction (0.078ms → 0.065ms)
- **Cause**: More efficient on-chip memory utilization, reducing memory access latency

### MoE Model Results
- **TPS Improvement**: 31% increase (10,200 → 13,400)
- **TPOT Reduction**: 23% reduction (0.098ms → 0.075ms)
- **Significance**: Larger gains for complex models with irregular computation patterns due to expert routing
- **Benefit**: Mitigates communication overheads that typically limit MoE model scaling

### Baseline Limitations
- **TP=8, PP=2**: Effective but does not consider on-chip memory constraints explicitly
- **Consequence**: More off-chip memory accesses and communication delays compared to proposed method