# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Configuration
- **Platform**: 16 NVIDIA H100 GPUs
- **Interconnect**: NVLink and NVSwitch
- **Setting**: Inference-only

### Model Architecture
- **Model Type**: Dense Transformer
- **Layers**: 4 layers
- **Architecture**: Standard feed-forward

### Fixed Parameters
- **Precision**: FP16
- **Batch Size**: 1024 tokens (fixed)
- **Attention Heads**: 16 heads
- **Head Dimension**: 512 per head
- **MLP Hidden Size**: 32768

### Baseline Configuration
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2
- **Note**: Without sequence parallelism or ring-based attention communication

## Evaluation Metrics

### Primary Metrics
1. **TPS (Tokens Per Second)**
   - Definition: Raw throughput of tokens processed per second
   - Interpretation: Higher is better

2. **TPOT (Time Per Output Token)**
   - Definition: Average latency per output token
   - Unit: Milliseconds
   - Interpretation: Lower is better

## Results

### Performance Comparison Table

| Model      | Method                | TPS (tokens/s) | TPOT (ms) |
|------------|-----------------------|----------------|-----------|
| Dense (4L) | Baseline (TP=8, PP=2) | 1.20M          | 0.85      |
| Dense (4L) | RA+SP                 | **1.45M**      | **0.70**  |

### Performance Improvements

#### Dense Model Results
- **TPS Improvement**: 20.8% increase (1.20M → 1.45M tokens/s)
- **TPOT Reduction**: 17.6% decrease (0.85 → 0.70 ms)
- **Dual Benefit**: Both higher throughput and reduced latency achieved

## Analysis

### Performance Drivers
1. **Ring-based Communication Pattern**
   - Avoids peak bandwidth demands of all-to-all exchanges
   - Reduces synchronization overhead

2. **Memory Savings from Sequence Parallelism**
   - Reduces activation footprint
   - Improves kernel scheduling efficiency

### Scalability Implications
- Benefits particularly significant for scenarios with high sequence length
- Performance improvements grow with model size and sequence length
- Effective utilization of distributed hardware resources

### Technical Validation
- Consistent performance benefits across tested configurations
- Method successfully addresses both communication bottlenecks and memory constraints
- Demonstrates practical viability for large-scale transformer deployments

## Experimental Validity
- **Controlled Environment**: Fixed hardware (16×H100 GPUs)
- **Consistent Parameters**: FP16 precision, 1024 token batch size maintained
- **Clear Baseline**: Well-defined TP=8, PP=2 configuration for comparison
- **Measurable Outcomes**: Quantitative improvements in both throughput and latency metrics