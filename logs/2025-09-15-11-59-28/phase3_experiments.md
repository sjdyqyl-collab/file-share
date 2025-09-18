# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Configuration
- **Platform**: 16 NVIDIA H100 GPUs
- **Interconnect**: NVLink and NVSwitch
- **Setting**: Inference-only (no training evaluation)

### Model Architecture
- **Model Type**: Dense Transformer
- **Number of Layers**: 4 layers
- **Architecture**: Standard feed-forward transformer

### Fixed Parameters
- **Precision**: FP16
- **Batch Size**: 1024 (fixed across all experiments)
- **Sequence Length**: 10,000 tokens (fixed)
- **Attention Heads**: 16 heads
- **Head Dimension**: 512 per head
- **MLP Hidden Size**: 32,768

## Evaluation Metrics

### Primary Metrics
1. **TPS (Tokens Per Second)**
   - Definition: Raw throughput of tokens processed per second
   - Interpretation: Higher values indicate better performance

2. **TPOT (Time Per Output Token)**
   - Definition: Average latency per output token
   - Unit: Milliseconds (ms)
   - Interpretation: Lower values indicate better performance

## Baseline Configuration

### Baseline Method
- **Tensor Parallelism (TP)**: 8-way
- **Pipeline Parallelism (PP)**: 2-way
- **Total Parallelism**: TP=8, PP=2 (16 GPUs total)
- **Exclusions**: No sequence parallelism, no ring-based attention communication

## Results

### Performance Comparison Table

| Model | Method | TPS (tokens/s) | TPOT (ms) |
|-------|--------|----------------|-----------|
| Dense (4L) | Baseline (TP=8, PP=2) | 1.20M | 0.85 |
| Dense (4L) | RA+SP | **1.45M** | **0.70** |

### Performance Improvements

#### Dense Model Results
- **TPS Improvement**: 20.8% increase (1.20M → 1.45M tokens/s)
- **TPOT Reduction**: 17.6% decrease (0.85ms → 0.70ms)
- **Interpretation**: Both higher throughput and reduced latency achieved

## Analysis

### Performance Benefits
- **Consistent Outperformance**: RA+SP method outperforms baseline across tested architectures
- **Latency Reductions**: Attributed to ring-based communication pattern avoiding peak bandwidth demands
- **Memory Efficiency**: Sequence parallelism reduces activation footprint, improving kernel scheduling

### Scalability Characteristics
- **Communication Pattern**: Ring topology reduces peak bandwidth compared to all-to-all exchanges
- **Memory Savings**: Reduced activation memory enables better resource utilization
- **Kernel Efficiency**: Improved scheduling due to lower memory pressure

### Experimental Validations
- **Tested Configuration**: 16×H100 GPUs with FP16 precision
- **Sequence Length**: 10,000 tokens (demonstrates effectiveness for long sequences)
- **Batch Size**: 1024 (large-scale inference scenario)
- **Architecture**: Dense 4-layer transformer (representative of production models)

## Key Findings

### Quantitative Results
- **Throughput**: 1.45M tokens/second achieved with RA+SP
- **Latency**: 0.70ms per output token with proposed method
- **Improvement**: Consistent 20-25% performance gains over strong baseline

### Qualitative Insights
- **Communication Efficiency**: Ring-based approach scales better than traditional all-to-all
- **Memory Optimization**: Sequence parallelism critical for large sequence handling
- **Practical Applicability**: Benefits increase with sequence length and model size