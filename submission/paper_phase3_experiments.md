# Phase 3: Experiments Extraction

## Abstract (Retained from Original)
We present a novel parallelization strategy for Multi-Head Attention (MHA) in large-scale transformer models that combines Ring Attention with sequence parallelism. Our approach leverages the communication-efficient properties of the ring topology to distribute attention computation across devices, while sequence parallelism reduces memory footprint by splitting input sequences across workers. This design minimizes all-to-all communication overhead, enhances scalability for extremely long sequences, and enables efficient utilization of distributed hardware resources. Experimental analysis indicates that the proposed method achieves substantial throughput improvements compared to conventional data- and tensor-parallel approaches, particularly in scenarios with high sequence length and large model size.

## Experimental Setup

### 1. Hardware Configuration
- **Platform**: 16×NVIDIA H100 GPUs
- **Interconnect**: NVLink and NVSwitch
- **Setting**: Inference-only evaluation

### 2. Model Architectures Tested
- **Dense Transformer**: 4 layers, standard feed-forward architecture
- **Mixture-of-Experts (MoE)**: 4 layers, top-2 gating, 8 experts, capacity factor 1.25

### 3. Fixed Parameters
- **Precision**: FP16
- **Batch Size**: 1024 tokens (fixed)
- **Number of Heads**: 16 (fixed)
- **Head Dimension**: 512 (fixed)
- **MLP Hidden Size**: 32768 (fixed)
- **MoE Routing**: Expert routing performed locally to avoid unnecessary communication for inactive experts

### 4. Baseline Configuration
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2
- **Note**: Baseline does NOT use sequence parallelism or ring-based attention communication

## Evaluation Metrics

1. **TPS (Tokens Per Second)**
   - Definition: Raw throughput of tokens processed per second
   - Interpretation: Higher values indicate better performance

2. **TPOT (Time Per Output Token)**
   - Definition: Average latency per output token, measured in milliseconds
   - Interpretation: Lower values indicate better performance

## Results

### Performance Comparison Table
| Model      | Method                | TPS (tokens/s) | TPOT (ms) |
|------------|-----------------------|----------------|-----------|
| Dense (4L) | Baseline (TP=8, PP=2) | 1.20M          | 0.85      |
| Dense (4L) | RA+SP                 | **1.45M**      | **0.70**  |
| MoE (4L)   | Baseline (TP=8, PP=2) | 0.95M          | 1.05      |
| MoE (4L)   | RA+SP                 | **1.18M**      | **0.82**  |

## Analysis

### Performance Improvements
- **Dense Model**:
  - TPS improvement: **20.8%** (from 1.20M to 1.45M tokens/s)
  - TPOT reduction: **17.6%** (from 0.85ms to 0.70ms)
  - Shows both higher throughput and reduced latency

- **MoE Model**:
  - TPS improvement: **24.2%** (from 0.95M to 1.18M tokens/s)
  - TPOT reduction: **21.9%** (from 1.05ms to 0.82ms)
  - Reflects greater communication and memory benefits in expert-based architectures

### Key Factors for Improvements
1. **Ring-based Communication Pattern**: Avoids peak bandwidth demands of all-to-all exchanges
2. **Memory Savings**: Sequence parallelism reduces activation footprint
3. **Improved Kernel Scheduling**: Reduced memory improves kernel scheduling efficiency
4. **Architecture Benefits**: MoE models show greater improvements due to more severe communication bottlenecks and memory fragmentation

### Scalability Implications
- Performance benefits grow with sequence length ($L$) and number of devices ($P$)
- Particularly effective for sequences with $L > 16$k tokens
- Benefits are consistent across both dense and MoE architectures
- Greater improvements observed in MoE models due to their inherent communication challenges