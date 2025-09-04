# Phase 3: Experiments Extraction

## 1. Experimental Setup

### Hardware Configuration
- **Platform**: 16× NVIDIA H100 GPUs
- **Interconnect**: NVLink and NVSwitch
- **Setting**: Inference-only (no training)

### Model Architecture
- **Model Type**: Dense Transformer
- **Layers**: 4 layers
- **Architecture**: Standard feed-forward transformer
- **Attention Heads**: 16 heads
- **Head Dimension**: 512 per head
- **MLP Hidden Size**: 32768
- **Precision**: FP16

### Fixed Parameters
- **Batch Size**: 1024 tokens (fixed)
- **Sequence Length**: Variable (with L > 16k showing significant benefits)

### Baseline Configuration
- **Method**: Tensor Parallelism (TP) + Pipeline Parallelism (PP)
- **Tensor Parallelism**: TP = 8
- **Pipeline Parallelism**: PP = 2
- **Note**: No sequence parallelism or ring-based attention communication

### Test Method
- **Proposed Method**: Ring Attention + Sequence Parallelism (RA+SP)
- **Devices**: 16 GPUs total for both baseline and proposed method

## 2. Evaluation Metrics

### Primary Metrics
1. **TPS (Tokens Per Second)**
   - Definition: Raw throughput of tokens processed per second
   - Direction: Higher is better
   - Measurement: Total tokens processed / total time

2. **TPOT (Time Per Output Token)**
   - Definition: Average latency per output token
   - Unit: Milliseconds (ms)
   - Direction: Lower is better
   - Measurement: Total inference time / total output tokens

## 3. Results

### Performance Comparison Table
| Model | Method | TPS (tokens/s) | TPOT (ms) |
|--------|--------|----------------|-----------|
| Dense (4L) | Baseline (TP=8, PP=2) | 1.20M | 0.85 |
| Dense (4L) | RA+SP | **1.45M** | **0.70** |

### Quantitative Improvements
- **TPS Improvement**: +20.8% (from 1.20M to 1.45M tokens/s)
- **TPOT Reduction**: -17.6% (from 0.85ms to 0.70ms per token)
- **Throughput Gain**: Consistent across tested architectures

## 4. Analysis

### Performance Benefits
- **Communication Efficiency**: Ring-based communication avoids peak bandwidth demands of all-to-all exchanges
- **Memory Efficiency**: Sequence parallelism reduces activation footprint, improving kernel scheduling efficiency
- **Scalability**: Benefits increase with sequence length L and number of devices P
- **Threshold**: Significant benefits observed for L > 16k tokens

### Latency Reduction Factors
1. **Ring Communication Pattern**: Sequential peer-to-peer exchanges reduce synchronization overhead
2. **Memory Savings**: Reduced activation memory enables better kernel scheduling
3. **Overlap**: Communication-computation overlap via asynchronous operations

### Scalability Characteristics
- **Linear Scaling**: Performance benefits grow with P (number of devices)
- **Sequence Length Dependency**: More pronounced benefits for longer sequences
- **Memory Scaling**: Activation memory scales as O(L/P) instead of O(L)

### Limitations and Considerations
- **Inference-Only**: Current evaluation limited to inference scenarios
- **Training Extension**: Future work needed for gradient communication in training
- **Hierarchical Topologies**: Potential for combining ring-based intra-node with bandwidth-aware inter-node scheduling
- **Precision Impact**: Mixed-precision (fp16/bf16) used to reduce bandwidth requirements