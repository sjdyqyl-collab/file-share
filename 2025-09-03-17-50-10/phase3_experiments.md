# Phase 3: Experiments Extraction

## 1. Experimental Setup

**Hardware Platform**:
- 16× NVIDIA H100 GPUs
- Interconnected via NVLink and NVSwitch

**Model Architecture**:
- **Dense Transformer**: 4 layers, standard feed-forward architecture
- Fixed parameters:
  - Precision: FP16
  - Batch size: 1024 tokens
  - Number of heads: 16
  - Dimension per head: 512
  - MLP hidden size: 32768

**Tested Method**:
- **RA+SP**: Ring Attention + Sequence Parallelism

**Baseline Configuration**:
- **Tensor Parallelism (TP) = 8**
- **Pipeline Parallelism (PP) = 2**
- **No sequence parallelism or ring-based attention communication**

## 2. Evaluation Metrics

1. **TPS (Tokens Per Second)**
   - Raw throughput of tokens processed per second
   - Higher values indicate better performance

2. **TPOT (Time Per Output Token)**
   - Average latency per output token in milliseconds
   - Lower values indicate better performance

## 3. Results

**Table 1 — Inference performance on 16×H100 GPUs (FP16, batch size 1024)**

| Model      | Method                | TPS (tokens/s) | TPOT (ms) |
|------------|-----------------------|----------------|-----------|
| Dense (4L) | Baseline (TP=8, PP=2) | 1.20M          | 0.85      |
| Dense (4L) | RA+SP                 | **1.45M**      | **0.70**  |

## 4. Performance Analysis

**Dense Model Results**:
- **TPS improvement**: 20.8% (from 1.20M to 1.45M tokens/s)
- **TPOT reduction**: 17.6% (from 0.85ms to 0.70ms per token)
- **Dual benefit**: Both higher throughput and reduced latency achieved

**Performance Drivers**:
1. **Ring-based communication pattern**: Avoids peak bandwidth demands of all-to-all exchanges
2. **Memory savings from sequence parallelism**: Reduces activation footprint
3. **Improved kernel scheduling efficiency**: Due to reduced memory pressure

**Scalability Characteristics**:
- Benefits grow with sequence length L and number of devices P
- Particularly effective for L > 16k tokens
- Consistent improvements across model architectures