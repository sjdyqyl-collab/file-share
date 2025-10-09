# Phase 3: Experiments Extraction - FA Pool Paper

## 4. Experimental Setup

### 4.1 Model Configuration
- **Architecture**: 4-layer Dense model
- **Parameters**: ~13B parameters
- **Hidden Dimension**: 4096
- **Attention Heads**: 32
- **Feed-forward Dimension**: 16384
- **Activation**: GELU
- **Normalization**: Pre-norm with RMSNorm

### 4.2 Baseline Configuration
- **Tensor Parallelism (TP)**: 8-way
- **Pipeline Parallelism (PP)**: 2-way
- **Total GPUs**: 16 GPUs (8×2 configuration)
- **Strategy**: Static allocation

### 4.3 FA Pool Configuration
- **Base Layer GPUs**: 8 GPUs (fixed)
- **Attention Pool**: Up to 32 additional GPUs (dynamic)
- **Sequence Threshold**: 4096 tokens
- **Maximum Pool Size**: 32 GPUs
- **Total Maximum GPUs**: 40 (8 base + 32 pool)

### 4.4 Evaluation Metrics
- **Time Per Output Token (TPOT)**: Average time per output token (milliseconds)
- **Tokens Per Second (TPS)**: Total tokens processed per second
- **Resource Utilization**: GPU utilization percentage
- **Memory Usage**: Per GPU memory consumption

### 4.5 Test Sequences
- **Short**: 512-2048 tokens
- **Medium**: 2048-8192 tokens
- **Long**: 8192-32768 tokens
- **Very Long**: 32768+ tokens

### 4.6 Hardware Configuration
- **GPU**: NVIDIA A100 80GB
- **Interconnect**: NVLink 3.0, InfiniBand
- **CPU**: AMD EPYC 7763
- **Memory**: 2TB DDR4
- **Storage**: NVMe SSD array

## 5. Results and Analysis

### 5.1 Overall Performance - TPOT Improvements
| Sequence Length | Baseline TPOT | FA Pool TPOT | Improvement |
|----------------|---------------|--------------|-------------|
| 512 tokens     | 45ms          | 41ms         | 1.1x        |
| 2048 tokens    | 78ms          | 56ms         | 1.4x        |
| 8192 tokens    | 245ms         | 117ms        | 2.1x        |
| 16384 tokens   | 892ms         | 279ms        | 3.2x        |

### 5.2 Overall Performance - TPS Improvements
| Sequence Length | Baseline TPS | FA Pool TPS | Improvement |
|----------------|--------------|-------------|-------------|
| 512 tokens     | 22.2         | 26.7        | 1.2x        |
| 2048 tokens    | 25.6         | 41.0        | 1.6x        |
| 8192 tokens    | 33.4         | 83.5        | 2.5x        |
| 16384 tokens   | 18.3         | 51.2        | 2.8x        |

### 5.3 Scaling Characteristics
- **Strong Scaling**: Near-linear scaling up to 16K tokens
- **Resource Utilization**: 85-92% (FA Pool) vs 45-60% (baseline)
- **Communication Overhead**: <15% of total computation time
- **Optimal Pool Size**: Performance plateaus beyond 24 GPUs

### 5.4 Memory Usage Analysis
- **Base Layer**: 65GB per GPU
- **Attention Pool**: 45GB per GPU (reduced due to block-wise computation)
- **Total Memory**: Comparable to baseline with better distribution
- **Memory Efficiency**: 20GB reduction per pool GPU due to Flash Attention

### 5.5 Overhead Analysis
| Component         | Percentage | Notes                          |
|-------------------|------------|--------------------------------|
| Attention Computation | 75-80%   | Improved from 85-90% baseline  |
| Communication     | 10-15%     | Optimized through hierarchical reduction |
| Synchronization   | 5-8%       | Minimized through async execution |
| Resource Management | 2-3%     | Efficient allocation/deallocation |

### 5.6 Resource Allocation Patterns
- **Threshold Effect**: Clear improvement at 4096+ tokens
- **Dynamic Adaptation**: Effective for varying sequence lengths
- **GPU Utilization**: 85-92% average utilization in attention pool
- **Optimal Configuration**: 24 GPUs for attention pool (plateau point)

### 5.7 Comparison with Static Strategies
- **vs TP=16, PP=2**: 2.1x better TPOT for 8K sequences
- **vs TP=8, PP=4**: 1.8x improvement in TPS for long sequences
- **Resource Efficiency**: Better utilization despite more total GPUs
- **Memory Overhead**: Lower per-GPU memory requirements

### 5.8 Limitations Identified
- **Communication Bottleneck**: Dominates for sequences >32K tokens
- **Memory Requirements**: Total system memory increases with pool size
- **Hardware Dependency**: Requires flexible GPU allocation capabilities
- **Energy Consumption**: Additional GPUs increase power usage

## 6. Key Findings

### 6.1 Performance Scaling
- **Linear Scaling**: Up to 16K tokens with 24 GPUs
- **Diminishing Returns**: Beyond 24 GPUs in attention pool
- **Threshold Validation**: 4096 token threshold empirically optimal
- **Long Sequence Advantage**: Greater improvements for longer sequences

### 6.2 Resource Utilization
- **Efficiency Gains**: 40-47% improvement in GPU utilization
- **Dynamic Allocation**: Adapts to varying computational demands
- **Communication Optimization**: Hierarchical reduction minimizes overhead
- **Memory Distribution**: Balanced across base and pool GPUs

### 6.3 Practical Implications
- **Deployment Flexibility**: Effective for variable sequence workloads
- **Cost-Benefit**: Justified for long sequence processing scenarios
- **Hardware Requirements**: Needs dynamic GPU allocation support
- **Energy Trade-offs**: Performance vs power consumption balance