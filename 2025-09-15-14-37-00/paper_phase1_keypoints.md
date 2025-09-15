# Paper Keypoints - Phase 1

## Abstract (Retained as-is)
In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.

## Key Contributions
1. **Layer-wise partitioning strategy** that ensures each partition fits within SRAM/L2 cache
2. **Memory footprint estimation method** for weights, activations, and temporary buffers
3. **Greedy and dynamic programming algorithms** for optimal partitioning
4. **20% improvement in TPS** over baseline (TP=8, PP=2) for dense model
5. **17% reduction in TPOT** compared to baseline

## Core Problem
- Large models require external memory access, causing latency and bandwidth bottlenecks
- Need to fit model partitions entirely within fast on-chip memory (SRAM/L2 cache)
- Must preserve execution order while minimizing partitions

## Method Summary
- **Input**: Model with *n* layers, cache capacity *C* per device
- **Output**: *k* partitions where each partition size ≤ *C*
- **Constraint**: Layers assigned contiguously in original order
- **Memory calculation**: size(layer) = weight_size + activation_size + buffer_size

## Experimental Setup
- **Hardware**: 16 NVIDIA H100 GPUs
- **Model**: 16-layer dense network
- **Configuration**: FP16, batch=1024, seq_len=10000, heads=16, head_dim=512, MLP_hidden=32768
- **Baseline**: TP=8, PP=2 (16 GPUs total)
- **Metrics**: Tokens Per Second (TPS), Time Per Output Token (TPOT)

## Results Summary
| Method | GPUs | TPS (tokens/s) | TPOT (ms) |
|--------|------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 |
| Proposed Layer-wise | 16 | 15,360 | 0.065 |

## Technical Specifications
- **Memory components**: Weights (FP16=2 bytes/param), Activations (feature_map × batch), Buffers (operator workspace)
- **Partitioning algorithms**: Greedy layer aggregation, Dynamic programming for balanced partitions
- **Edge cases**: Single layer > cache size requires compression/intra-layer partitioning