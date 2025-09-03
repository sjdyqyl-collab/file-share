# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Platform
- 16 NVIDIA H100 GPUs
- FP16 precision
- Batch size: 1024

### Model Specifications
- **Dense model**: 16-layer fully connected dense network
- **Head configuration**: 16 heads, 512 dimension per head
- **MLP hidden size**: 32,768

### Baseline Comparison
- **Baseline method**: Tensor Parallelism (TP) + Pipeline Parallelism (PP)
- **Baseline configuration**: TP=8, PP=2 (fully utilizes 16 GPUs: 8×2=16)

### Performance Metrics
- **TPS (Tokens Per Second)**: Number of output tokens generated per second
- **TPOT (Time Per Output Token)**: Average time to produce single output token (milliseconds)

## Results

| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

## Analysis
- **20% increase in TPS** (15,360 vs 12,800)
- **17% reduction in TPOT** (0.065ms vs 0.078ms)
- Improvement attributed to efficient on-chip memory utilization
- Baseline approach doesn't explicitly consider on-chip memory constraints, leading to more off-chip memory accesses and communication delays