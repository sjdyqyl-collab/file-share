# Phase 3: Experiments Extraction

## 1. Experimental Setup

### Hardware Configuration
- **GPUs**: 16 NVIDIA H100 GPUs
- **Interconnect**: NVLink and NVSwitch
- **Total devices**: P = 16

### Model Architecture
- **Model Type**: Dense Transformer
- **Layers**: 4 layers
- **Architecture**: Standard feed-forward transformer

### Fixed Parameters
- **Precision**: FP16
- **Batch size**: 1024 tokens (fixed)
- **Number of attention heads**: 16 (fixed)
- **Head dimension**: 512 (fixed)
- **MLP hidden size**: 32768 (fixed)
- **Model hidden size**: 8192 (calculated as 16 heads × 512 dimensions)

### Baseline Configuration
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2
- **Sequence Parallelism**: None
- **Ring Attention**: None

### Proposed Configuration
- **Method**: Ring Attention + Sequence Parallelism (RA+SP)
- **Sequence Parallelism**: Active (split across 16 devices)
- **Ring Attention**: Active (16-stage ring)

## 2. Evaluation Metrics

### Primary Metrics
1. **TPS (Tokens Per Second)**
   - Definition: Raw throughput of tokens processed per second
   - Optimization goal: Higher is better
   - Baseline measurement unit: tokens/second

2. **TPOT (Time Per Output Token)**
   - Definition: Average latency per output token
   - Measurement unit: milliseconds (ms)
   - Optimization goal: Lower is better

## 3. Experimental Results

### Dense Transformer (4L) Performance Comparison

| Model Configuration | Method | TPS (tokens/s) | TPOT (ms) |
|-------------------|----------|----------------|-----------|
| Dense (4 layers) | Baseline (TP=8, PP=2) | 1.20M | 0.85 |
| Dense (4 layers) | **RA+SP** | **1.45M** | **0.70** |

### Performance Improvements
- **TPS Improvement**: 20.8% increase (1.45M vs 1.20M tokens/s)
- **TPOT Reduction**: 17.6% decrease (0.70ms vs 0.85ms)
- **Latency Reduction**: 24-27% improvement range
- **Throughput Gain**: 20-25% improvement range

## 4. Analysis and Insights

### Performance Benefits
1. **Communication Efficiency**: Ring-based communication pattern avoids peak bandwidth demands of all-to-all exchanges
2. **Memory Efficiency**: Sequence parallelism reduces activation footprint, improving kernel scheduling efficiency
3. **Scalability**: Benefits grow with sequence length (L) and number of devices (P), particularly effective for L > 16k tokens
4. **Overlap**: Computation and communication overlap achieved through ring topology

### Key Factors for Improvement
- **Reduced Communication Overhead**: Ring topology eliminates all-to-all communication
- **Memory Footprint Reduction**: Activation memory reduced by factor of P (16x in this setup)
- **Bandwidth Utilization**: Lower peak bandwidth requirements enable better hardware utilization
- **Kernel Efficiency**: Reduced memory pressure improves kernel scheduling and execution efficiency

### Limitations and Considerations
- **Inference-only setting**: Results demonstrated for inference, training scenarios require additional considerations
- **Hierarchical topology**: Future work includes combining intra-node and inter-node communication
- **Precision impact**: Mixed-precision (fp16/bf16) used to reduce bandwidth requirements
- **Sequence length dependency**: Benefits more pronounced for longer sequences (>16k tokens)