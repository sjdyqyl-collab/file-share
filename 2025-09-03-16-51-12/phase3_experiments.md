# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Configuration
- **Platform**: 16 NVIDIA H100 GPUs
- **Interconnect**: NVLink and NVSwitch
- **Setting**: Inference-only evaluation

### Model Architecture
- **Model Type**: Dense Transformer
- **Layers**: 4 layers
- **Architecture**: Standard feed-forward transformer
- **Precision**: FP16

### Fixed Parameters
- **Batch Size**: 1024 tokens
- **Number of Heads**: 16
- **Head Dimension**: 512
- **MLP Hidden Size**: 32768

### Baseline Configuration
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2
- **Note**: Baseline does NOT use sequence parallelism or ring-based attention communication

## Evaluation Metrics

### Primary Metrics
1. **TPS (Tokens Per Second)**
   - Definition: Raw throughput of tokens processed per second
   - Interpretation: Higher values indicate better performance

2. **TPOT (Time Per Output Token)**
   - Definition: Average latency per output token
   - Unit: Milliseconds (ms)
   - Interpretation: Lower values indicate better performance

## Results

### Performance Comparison Table
| Model | Method | TPS (tokens/s) | TPOT (ms) |
|-------|--------|----------------|-----------|
| Dense (4L) | Baseline (TP=8, PP=2) | 1.20M | 0.85 |
| Dense (4L) | RA+SP (Ring Attention + Sequence Parallelism) | **1.45M** | **0.70** |

### Performance Improvements
- **TPS Improvement**: 20.8% increase (from 1.20M to 1.45M tokens/s)
- **TPOT Reduction**: 17.6% decrease (from 0.85ms to 0.70ms per token)
- **Dual Benefit**: Both higher throughput and reduced latency achieved

## Analysis

### Performance Drivers
1. **Ring-based Communication Pattern**
   - Avoids peak bandwidth demands of all-to-all exchanges
   - Reduces synchronization overhead

2. **Memory Savings from Sequence Parallelism**
   - Reduces activation footprint
   - Improves kernel scheduling efficiency

### Scalability Characteristics
- Benefits particularly significant for sequences longer than 16k tokens
- Performance improvements scale with sequence length and device count
- Effective in memory-constrained environments

### Implementation Validation
- Method successfully implemented on 16×H100 GPU cluster
- Consistent performance gains across dense transformer architecture
- Demonstrates practical viability for large-scale deployment