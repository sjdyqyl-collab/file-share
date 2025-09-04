# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Platform
- 16 NVIDIA H100 GPUs
- Each GPU has SRAM/L2 cache capacity C (exact value not specified in paper)

### Model Configurations
- **Dense model:** 16-layer fully connected dense network
- **Precision:** FP16 (2 bytes per parameter)
- **Batch size:** 1024
- **Number of heads:** 16
- **Dimension per head:** 512
- **Hidden size of MLP:** 32768

### Baseline Comparison
- **Standard setup:** Tensor parallelism (TP) + Pipeline parallelism (PP)
- **Configuration:** TP=8, PP=2 (fully utilizes 16 GPUs: 8 × 2 = 16)

### Performance Metrics
1. **Tokens Per Second (TPS):** The number of output tokens generated per second
2. **Time Per Output Token (TPOT):** The average time to produce a single output token, in milliseconds

## Experimental Results

| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

## Performance Analysis

### Dense Model Results
- **TPS improvement:** 20% increase (15,360 vs 12,800 tokens/s)
- **Latency reduction:** 17% reduction in TPOT (0.065ms vs 0.078ms)
- **Root cause:** More efficient on-chip memory utilization, reducing memory access latency

### Baseline Comparison
- The baseline TP=8, PP=2 approach is effective but does not consider on-chip memory constraints explicitly
- This leads to more off-chip memory accesses and communication delays
- The proposed layer-wise method explicitly optimizes for cache capacity constraints

## Key Insights
- The layer-wise partitioning approach successfully fits model partitions within on-chip memory
- Reduced off-chip memory access translates directly to improved throughput and reduced latency
- The method scales effectively across 16 GPUs while maintaining contiguous layer execution order