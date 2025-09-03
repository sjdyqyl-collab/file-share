# Phase 3: Experiments Extraction

## Experiments

### Setup
- **Stage**: Inference stage evaluation
- **Hardware Platform**: 16 NVIDIA H100 GPUs
- **Model Types**:
  - Dense model: 16-layer fully connected dense network
- **Precision**: FP16 (16-bit floating point)
- **Batch Size**: 1024
- **Model Configuration**:
  - Number of heads: 16 (fixed)
  - Dimension of each head: 512 (fixed)
  - Hidden size of MLP: 32768 (fixed)
- **Baseline Comparison**: Standard tensor parallelism (TP) and pipeline parallelism (PP) setup
  - TP=8, PP=2
  - Fully utilizes 16 GPUs (8 × 2 = 16)

### Performance Metrics
- **TPS (Tokens Per Second)**: Number of output tokens generated per second
- **TPOT (Time Per Output Token)**: Average time to produce a single output token, measured in milliseconds

### Results Table
| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

### Analysis
- **Dense model performance**: 
  - 20% increase in TPS (from 12,800 to 15,360 tokens/s)
  - 17% reduction in TPOT (from 0.078 to 0.065 ms)
- **Improvement source**: More efficient on-chip memory utilization, reducing memory access latency
- **Baseline limitation**: TP=8, PP=2 approach does not consider on-chip memory constraints explicitly, leading to more off-chip memory accesses and communication delays

## Conclusion Summary
- Proposed layer-wise deployment strategy partitions model layers across multiple accelerator cards
- Each partition fits entirely within SRAM/L2 cache of target hardware
- Significantly reduces off-chip memory accesses and improves inference efficiency
- Experimental results demonstrate substantial performance gains over baseline tensor and pipeline parallelism
- Future work includes extending to training workloads and adaptive partitioning strategies