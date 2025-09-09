# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Configuration
- Platform: 16 NVIDIA H100 GPUs
- Memory hierarchy per GPU:
  - SRAM/L2 cache capacity: To be calculated from layer constraints
  - DRAM: HBM3 memory
- Interconnect: NVLink/PCIe for inter-GPU communication

### Model Configuration
- **Dense Model**: 16-layer fully connected network
- **Precision**: FP16 (2 bytes per value)
- **Batch Size**: 1024
- **Architecture Parameters**:
  - Number of attention heads: 16
  - Dimension per head: 512
  - Hidden size: 16 × 512 = 8192
  - MLP intermediate size: 32768
  - Total parameters per layer: Calculated below

### Memory Calculation for Dense Model

#### Per Layer Memory Footprint
1. **Attention Layer**:
   - Q/K/V projections: 3 × (hidden_size × hidden_size) = 3 × 8192 × 8192 = 201,326,592 parameters
   - Output projection: hidden_size × hidden_size = 8192 × 8192 = 67,108,864 parameters
   - Attention weights total: 268,435,456 parameters × 2 bytes = 536,870,912 bytes

2. **MLP Layer**:
   - First linear: hidden_size × MLP_size = 8192 × 32768 = 268,435,456 parameters
   - Second linear: MLP_size × hidden_size = 32768 × 8192 = 268,435,456 parameters
   - MLP weights total: 536,870,912 parameters × 2 bytes = 1,073,741,824 bytes

3. **Total Per Layer**:
   - Weights: 805,306,368 bytes ≈ 805 MB
   - Activations (batch_size × hidden_size): 1024 × 8192 × 2 bytes = 16,777,216 bytes ≈ 16 MB
   - Buffers: ~10% of weights ≈ 80 MB
   - **Total per layer**: ~901 MB

#### Cache Capacity Requirement
- For 16 layers with equal distribution: 901 MB × (16/k) ≤ C
- With k=16 partitions: 901 MB per layer fits in cache
- Therefore, SRAM/L2 cache capacity C ≥ 901 MB

### Baseline Configuration
- **Tensor Parallelism (TP)**: 8-way
- **Pipeline Parallelism (PP)**: 2-way
- **Total GPUs**: 8 × 2 = 16 GPUs
- **Distribution**:
  - 8 GPUs handle tensor-parallel portions of each layer
  - 2 pipeline stages across 16 GPUs total

### Proposed Configuration
- **Layer-wise Partitioning**: 16 partitions
- **Distribution**: 1 layer per GPU across 16 GPUs
- **Cache Constraint**: Each layer (901 MB) fits in SRAM/L2 cache
- **Communication**: Only between adjacent layers on different GPUs

### Performance Results

| Configuration | GPUs | TPS (tokens/s) | TPOT (ms) | Improvement |
|---------------|------|----------------|-----------|-------------|
| Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 | - |
| Proposed Layer-wise | 16 | 15,360 | 0.065 | +20% TPS, -17% TPOT |

### Analysis
- **20% throughput improvement** from reduced off-chip memory access
- **17% latency reduction** due to cache locality
- Baseline suffers from inter-GPU communication overhead in tensor parallelism
- Proposed method minimizes communication to layer boundaries only