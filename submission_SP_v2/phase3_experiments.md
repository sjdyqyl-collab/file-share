# Phase 3: Experiments Extraction

## Experimental Setup

### 1. Hardware Configuration
- **GPUs**: 16 NVIDIA H100 GPUs
- **Interconnect**: NVLink and NVSwitch
- **Setting**: Inference-only evaluation

### 2. Model Architectures

#### Dense Transformer
- **Layers**: 4 layers
- **Architecture**: Standard feed-forward transformer
- **Attention Heads**: 16 heads
- **Head Dimension**: 512 per head
- **Hidden Size**: Calculated as 16 × 512 = 8192
- **MLP Hidden Size**: 32768

#### Mixture-of-Experts (MoE) Transformer
- **Layers**: 4 layers
- **Expert Configuration**: Top-2 gating with 8 experts total
- **Capacity Factor**: 1.25
- **Attention Heads**: 16 heads
- **Head Dimension**: 512 per head
- **MLP Hidden Size**: 32768
- **Expert Routing**: Performed locally to avoid communication for inactive experts

### 3. Fixed Parameters
- **Precision**: FP16
- **Batch Size**: 1024 tokens (fixed)
- **Sequence Length**: >16k tokens for optimal benefits

### 4. Baseline Configuration
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2
- **Sequence Parallelism**: Not used
- **Ring-based Attention**: Not used

## Evaluation Metrics

### 1. TPS (Tokens Per Second)
- **Definition**: Raw throughput of tokens processed per second
- **Direction**: Higher is better
- **Measurement**: Total tokens processed divided by total time

### 2. TPOT (Time Per Output Token)
- **Definition**: Average latency per output token
- **Unit**: Milliseconds (ms)
- **Direction**: Lower is better
- **Measurement**: Total inference time divided by number of output tokens

## Results

### Performance Comparison Table

| Model Type | Method Configuration | TPS (tokens/s) | TPOT (ms) | Improvement |
|------------|---------------------|----------------|-----------|-------------|
| Dense (4L) | Baseline (TP=8, PP=2) | 1.20M | 0.85 | - |
| Dense (4L) | RA+SP | **1.45M** | **0.70** | +20.8% TPS, -17.6% TPOT |
| MoE (4L) | Baseline (TP=8, PP=2) | 0.95M | 1.05 | - |
| MoE (4L) | RA+SP | **1.18M** | **0.82** | +24.2% TPS, -21.9% TPOT |

## Analysis

### 1. Dense Model Performance
- **TPS Improvement**: 20.8% increase (1.20M → 1.45M tokens/s)
- **Latency Reduction**: 17.6% decrease (0.85ms → 0.70ms per token)
- **Key Factors**: Ring-based communication pattern reduces peak bandwidth demands

### 2. MoE Model Performance
- **TPS Improvement**: 24.2% increase (0.95M → 1.18M tokens/s)
- **Latency Reduction**: 21.9% decrease (1.05ms → 0.82ms per token)
- **Key Factors**: Greater benefits due to communication bottlenecks and memory fragmentation in expert-based architectures

### 3. Communication Benefits
- **Peak Bandwidth**: Ring topology avoids peak bandwidth demands of all-to-all exchanges
- **Memory Savings**: Sequence parallelism reduces activation footprint
- **Kernel Efficiency**: Improved kernel scheduling due to reduced memory pressure

### 4. Scalability Observations
- **Sequence Length**: Benefits grow with L > 16k tokens
- **Device Count**: Performance improvements scale with number of devices P
- **Architecture**: MoE models show greater relative improvements than dense models

## Experimental Methodology

### 1. Measurement Process
- **Warm-up**: Multiple warm-up runs to stabilize GPU performance
- **Averaging**: Results averaged over multiple runs
- **Isolation**: Network and compute measurements isolated

### 2. Reproducibility Details
- **Random Seeds**: Fixed for consistent expert routing in MoE
- **Data**: Synthetic data with controlled sequence lengths
- **Environment**: Controlled temperature and power settings on H100 GPUs