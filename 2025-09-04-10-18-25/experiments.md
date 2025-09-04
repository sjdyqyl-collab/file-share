# Experiments: Ring Attention + Sequence Parallelism

## 1. Experimental Setup

**Hardware Configuration:**
- Platform: 16 NVIDIA H100 GPUs
- Interconnect: NVLink and NVSwitch
- Mode: Inference-only setting

**Model Architecture:**
- **Dense Transformer**
  - Layers: 4
  - Architecture: Standard feed-forward transformer
  - Attention heads: 16
  - Head dimension: 512
  - MLP hidden size: 32,768
  - Model hidden size: 8,192 (16 heads × 512 dimensions)

**Fixed Parameters:**
- Precision: FP16
- Batch size: 1,024 tokens
- Number of heads: 16 (fixed)
- Head dimension: 512 (fixed)
- MLP hidden size: 32,768 (fixed)

## 2. Baseline Configuration

**Baseline Method:**
- Tensor Parallelism (TP): 8
- Pipeline Parallelism (PP): 2
- **No sequence parallelism or ring-based attention communication**
- Total devices: 16 (TP×PP = 8×2 = 16)

## 3. Proposed Method Configuration

**RA+SP Method:**
- Ring Attention + Sequence Parallelism
- Devices: 16 (arranged in logical ring)
- Sequence split across 16 devices
- Ring communication with 16 stages

## 4. Evaluation Metrics

1. **TPS (Tokens Per Second)**
   - Definition: Raw throughput of tokens processed per second
   - Higher values indicate better performance

2. **TPOT (Time Per Output Token)**
   - Definition: Average latency per output token in milliseconds
   - Lower values indicate better performance

## 5. Results Table

**Table 1 — Inference performance on 16×H100 GPUs (FP16, batch size 1024)**

| Model      | Method                | TPS (tokens/s) | TPOT (ms) |
|------------|-----------------------|----------------|-----------|
| Dense (4L) | Baseline (TP=8, PP=2) | 1.20M          | 0.85      |
| Dense (4L) | RA+SP                 | **1.45M**      | **0.70**  |

## 6. Performance Analysis

**Dense Model Results:**
- **TPS Improvement:** 20.8% (1.45M vs 1.20M)
- **TPOT Reduction:** 17.6% (0.70ms vs 0.85ms)
- **Combined Benefits:** Higher throughput and reduced latency

## 7. Performance Drivers

**Latency Reduction Factors:**
1. **Ring-based communication pattern:** Avoids peak bandwidth demands of all-to-all exchanges
2. **Memory savings from sequence parallelism:** Reduces activation footprint
3. **Improved kernel scheduling efficiency:** Due to reduced memory pressure

**Scalability Benefits:**
- Performance improvements grow with sequence length (L) and number of devices (P)
- Particularly effective for L > 16k tokens
- Benefits especially significant for memory-constrained environments

## 8. Communication Pattern Comparison

**Baseline (TP=8, PP=2):**
- All-to-all communication patterns
- Higher peak bandwidth requirements
- Tensor parallelism across 8 devices per layer
- Pipeline parallelism across 2 stages

**RA+SP:**
- Ring topology communication
- Lower peak bandwidth requirements
- Sequential peer-to-peer exchanges
- 16 stages of ring communication
- Sequence dimension split across all 16 devices