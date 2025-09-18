# Phase 3: Experiments Extraction

## 1. Experimental Setup

### Hardware Configuration
- **Platform**: 16×NVIDIA H100 GPUs
- **Interconnect**: NVLink and NVSwitch
- **Setting**: Inference-only evaluation

### Model Architecture
- **Model**: Dense Transformer
- **Layers**: 4 layers
- **Architecture**: Standard feed-forward transformer

### Fixed Parameters
- **Precision**: FP16
- **Batch Size**: 1024
- **Sequence Length**: 10,000 tokens
- **Attention Heads**: 16 heads
- **Head Dimension**: 512 per head
- **MLP Hidden Size**: 32,768

### Parallelism Configurations

#### Baseline Configuration
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2
- **Features**: Without sequence parallelism or ring-based attention communication

#### Proposed Configuration (RA+SP)
- **Ring Attention + Sequence Parallelism (RA+SP)**
- **Devices**: 16 (full utilization)
- **Ring Size**: 16 devices in logical ring
- **Sequence Split**: 10,000/16 = 625 tokens per device

## 2. Evaluation Metrics

### Primary Metrics
1. **TPS (Tokens Per Second)**
   - Definition: Raw throughput of tokens processed per second
   - Interpretation: Higher is better

2. **TPOT (Time Per Output Token)**
   - Definition: Average latency per output token in milliseconds
   - Interpretation: Lower is better

## 3. Results Summary

### Performance Comparison Table
| Model | Method | TPS (tokens/s) | TPOT (ms) |
|-------|--------|----------------|-----------|
| Dense (4L) | Baseline (TP=8, PP=2) | 1.20M | 0.85 |
| Dense (4L) | RA+SP | **1.45M** | **0.70** |

### Performance Improvements
- **TPS Improvement**: 20.8% increase (1.45M vs 1.20M tokens/sec)
- **Latency Reduction**: 17.6% decrease (0.70ms vs 0.85ms TPOT)

## 4. Analysis Details

### Performance Benefits
- **Throughput**: Consistent 20-25% higher TPS across configurations
- **Latency**: 24-27% better TPOT performance
- **Scalability**: Benefits increase with sequence length and device count

### Technical Advantages
- **Communication**: Ring-based pattern avoids peak bandwidth demands of all-to-all exchanges
- **Memory**: Sequence parallelism reduces activation footprint
- **Scheduling**: Improved kernel scheduling efficiency due to reduced memory pressure

### Contextual Performance
- **Sequence Length**: Benefits particularly significant for L > 16k tokens
- **Device Scaling**: Performance improvements grow with P (number of devices)
- **Memory Efficiency**: Enables processing of longer sequences within memory constraints

## 5. Experimental Validation

### Consistency Check
- Results validated across multiple runs
- Performance gains consistent across different sequence lengths (when tested)
- Memory usage verified to match theoretical predictions

### Bottleneck Analysis
- **Communication**: Ring topology successfully reduces peak bandwidth requirements
- **Memory**: Sequence parallelism effectively reduces per-device memory footprint
- **Compute**: Better overlap between communication and computation phases