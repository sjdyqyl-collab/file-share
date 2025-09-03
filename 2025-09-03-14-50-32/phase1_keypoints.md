# Phase 1: Key Points Extraction

## Core Problem
- Large neural networks require external memory access, causing latency and bandwidth bottlenecks
- Need to leverage fast on-chip memory (SRAM/L2 cache) for efficient deployment

## Key Innovation
- Novel layer-wise deployment strategy that partitions n layers across multiple accelerator cards
- Each partition must fit entirely within SRAM/L2 cache capacity C of individual device
- Minimizes off-chip memory accesses and reduces latency

## Technical Approach
1. **Problem Formulation**: Partition n layers into k disjoint groups P = {P1, P2, ..., Pk}
   - Each group Pi assigned to separate hardware accelerator
   - Memory footprint S(Pi) ≤ C (cache capacity)
   - Layers assigned contiguously in original order
   - Minimize number of partitions k

2. **Memory Footprint Calculation**: 
   - size(lj) = weight_size(lj) + activation_size(lj) + buffer_size(lj)
   - Weights: parameter tensors × datatype size (FP16 = 2 bytes)
   - Activations: output feature map dimensions × batch size
   - Buffers: workspace memory for operators

3. **Partitioning Algorithms**:
   - Greedy Layer Aggregation: Simple sequential grouping until capacity reached
   - Dynamic Programming: Optimize for balanced partitions (optional)

## Experimental Results
- **Hardware**: 16 NVIDIA H100 GPUs
- **Models**: 16-layer dense network
- **Precision**: FP16, batch size 1024
- **Configuration**: 16 heads, 512 head dimension, 32768 MLP hidden size
- **Baseline**: TP=8, PP=2 (tensor parallelism × pipeline parallelism)
- **Metrics**: TPS (tokens/second), TPOT (time per output token in ms)

## Performance Gains
- Dense model: 20% TPS increase (12,800 → 15,360 tokens/s)
- Dense model: 17% TPOT reduction (0.078 → 0.065 ms)
- Improvement from reduced memory access latency via on-chip cache utilization

## Key Constraints
- Single layer memory footprint must not exceed cache capacity C
- Batch size tuning can help fit activation constraints
- May require intra-layer partitioning or compression for oversized layers