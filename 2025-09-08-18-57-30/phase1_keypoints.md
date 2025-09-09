# Phase 1: Key Points Extraction

## Key Problem
- Large neural networks exceed on-chip memory (SRAM/L2 cache) capacity
- Off-chip memory access creates latency and bandwidth bottlenecks
- Need deployment strategy that maximizes on-chip memory utilization

## Key Innovation
- Layer-wise partitioning: Split n layers into k groups fitting cache capacity C
- Each partition P_i must satisfy: S(P_i) = Σ size(l_j) ≤ C
- Contiguous layer assignment preserving execution order

## Key Components
1. **Memory footprint estimation**: weights + activations + temporary buffers
2. **Partitioning algorithms**: Greedy layer aggregation or dynamic programming
3. **Deployment strategy**: Each partition on separate accelerator card
4. **Performance metrics**: TPS (tokens/second) and TPOT (time/token in ms)

## Key Results
- Dense 16-layer model: 20% TPS increase (12,800 → 15,360)
- 17% TPOT reduction (0.078ms → 0.065ms)
- Baseline: TP=8, PP=2 on 16 GPUs
- Proposed: Layer-wise on 16 GPUs

## Key Dimensions
- Model: 16 layers
- Precision: FP16 (2 bytes)
- Batch size: 1024
- Heads: 16
- Head dimension: 512
- MLP hidden size: 32768
- Hardware: 16 NVIDIA H100 GPUs