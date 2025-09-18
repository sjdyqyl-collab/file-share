# Experiments - Phase 3

## Experimental Setup

### Hardware Platform
- 16 × NVIDIA H100 GPUs

### Model Specifications
- **Dense Model**: 16-layer fully connected dense network
- **Precision**: FP16 (2 bytes per parameter)
- **Batch Size**: 1024
- **Head Configuration**: 16 heads, 512 dimensions per head
- **Hidden Size**: 8192 (16 × 512)
- **MLP Hidden Size**: 32768

### Baseline Configuration
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2
- **Total GPUs**: 8 × 2 = 16 (fully utilized)

### Performance Metrics
- **Tokens Per Second (TPS)**: Output tokens generated per second
- **Time Per Output Token (TPOT)**: Average time per output token (milliseconds)

## Results

| Model | Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------|--------|------|----------------|-----------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 |

## Performance Analysis
- **20% increase in TPS** (12,800 → 15,360 tokens/s)
- **17% reduction in TPOT** (0.078 → 0.065 ms)
- Improvement attributed to efficient on-chip memory utilization reducing memory access latency
- Baseline TP=8, PP=2 approach lacks explicit on-chip memory consideration, leading to more off-chip accesses and communication delays

## Memory Calculation Details
For deployment configuration, we need to determine cache capacity constraints:
- Each layer memory footprint = weights + activations + buffers
- Total model parameters: 16 layers × (attention + MLP components)
- With FP16 precision, each parameter = 2 bytes
- Activation memory scales with batch size (1024) and hidden dimensions
- Cache capacity C must accommodate complete partition Pᵢ