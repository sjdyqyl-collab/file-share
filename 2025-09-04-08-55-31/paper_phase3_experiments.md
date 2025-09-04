# Phase 3: Experiments Extraction

## 1. Experimental Setup

**Hardware Configuration:**
- Platform: 16 NVIDIA H100 GPUs
- Interconnect: NVLink and NVSwitch
- Setting: Inference-only (no training)

**Model Architecture:**
- **Dense Transformer**: 4 layers
- Standard feed-forward architecture
- Precision: FP16
- Batch size: 1024 tokens (fixed)
- Attention heads: 16 (fixed)
- Head dimension: 512 (fixed)
- MLP hidden size: 32768 (fixed)

**Baseline Configuration:**
- Tensor Parallelism (TP): 8
- Pipeline Parallelism (PP): 2
- No sequence parallelism
- No ring-based attention communication

## 2. Evaluation Metrics

**Primary Metrics:**
1. **TPS (Tokens Per Second)**
   - Definition: Raw throughput of tokens processed per second
   - Higher values indicate better performance
   - Measures overall system throughput

2. **TPOT (Time Per Output Token)**
   - Definition: Average latency per output token in milliseconds
   - Lower values indicate better performance
   - Measures per-token latency

## 3. Results

**Table 1: Inference Performance Comparison**
| Model | Method | TPS (tokens/s) | TPOT (ms) |
|-------|--------|----------------|-----------|
| Dense (4L) | Baseline (TP=8, PP=2) | 1.20M | 0.85 |
| Dense (4L) | RA+SP | **1.45M** | **0.70** |

## 4. Performance Analysis

**Improvements over Baseline:**
- **TPS Improvement**: 20.8% increase (1.20M → 1.45M tokens/s)
- **TPOT Reduction**: 17.6% decrease (0.85ms → 0.70ms)
- Both higher throughput and reduced latency achieved

**Key Factors for Improvement:**
1. **Ring-based Communication Pattern**
   - Avoids peak bandwidth demands of all-to-all exchanges
   - More efficient bandwidth utilization
   - Better overlap of communication and computation

2. **Memory Savings from Sequence Parallelism**
   - Reduced activation footprint
   - Improved kernel scheduling efficiency
   - Better memory bandwidth utilization

## 5. Experimental Conditions

**Fixed Parameters:**
- Precision: FP16 throughout
- Batch size: 1024 tokens (constant)
- Model size: 4-layer dense transformer
- Hardware: 16×H100 GPUs
- Evaluation: Inference-only setting

**Variable Factors:**
- Parallelization strategy: Baseline vs RA+SP
- Communication pattern: All-to-all vs ring-based
- Memory management: Full sequence vs sequence parallelism

## 6. Scalability Insights

**Performance Characteristics:**
- Benefits increase with sequence length (L) and number of devices (P)
- Particularly effective for L > 16k tokens
- Consistent improvements across tested configurations

**System-Level Impact:**
- Reduced memory pressure per device
- Improved GPU utilization
- Better scaling efficiency on distributed systems

## 7. Validation Details

**Reproducibility:**
- Fixed random seeds
- Consistent hardware configuration
- Multiple runs averaged for final metrics
- Standard deviation < 2% across runs

**Baseline Strength:**
- Strong baseline with TP=8, PP=2
- Represents state-of-the-art conventional approach
- Ensures meaningful improvement measurement