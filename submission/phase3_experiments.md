# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Configuration
- System: 16 NVIDIA H100 GPUs
- Precision: Mixed precision (FP16) for throughput and numerical stability
- Total devices: 16 (fully utilized)

### Model Architectures Tested
1. **2-layer Dense Transformer model**
2. **2-layer Mixture-of-Experts (MoE) Transformer model**
   - 4 experts per layer

### Fixed Model Parameters
- Number of heads: 16 (h=16)
- Dimension per head: 512 (d=512)
- Batch size: 1024
- Hidden size of MLP: 32768
- Total embedding dimension: D = h × d = 16 × 512 = 8192

## Baseline Configuration
- **Method**: Tensor Parallelism (TP) + Pipeline Parallelism (PP)
- **TP degree**: 8
- **PP degree**: 2
- **Total devices**: TP × PP = 8 × 2 = 16 GPUs
- **Description**: Widely adopted method for large-scale model deployment

## Evaluation Metrics
1. **Throughput (TPS)**: Tokens processed per second
2. **Time Per Output Token (TPOT)**: Average synchronization and communication overhead time per token (milliseconds)

## Experimental Results

### Dense Model Results
| Method | TPS (tokens/sec) | TPOT (ms) |
|--------|------------------|-----------|
| Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| Proposed (m×n=16) | 1,580,000 | 0.22 |
| **Improvement** | **+31.7%** | **-37.1%** |

### MoE Model Results
| Method | TPS (tokens/sec) | TPOT (ms) |
|--------|------------------|-----------|
| Baseline (TP=8, PP=2) | 850,000 | 0.45 |
| Proposed (m×n=16) | 1,150,000 | 0.30 |
| **Improvement** | **+35.3%** | **-33.3%** |

## Analysis of Results

### Performance Improvements
- **Dense model**: 31.7% throughput increase (1.2M → 1.58M tokens/sec)
- **MoE model**: 35.3% throughput increase (850K → 1.15M tokens/sec)
- **Communication overhead reduction**: 33-37% decrease in TPOT

### Hardware Utilization
- Proposed method fully exploits all 16 GPUs via m×n=16 partitions
- Direct mapping: each partition → one device
- Eliminates idle devices present in head-only partitioning when device count > head count

### Efficiency Gains
- **Load balancing**: Even distribution across both heads and dimensions
- **Memory footprint**: Each device stores fraction of MHA parameters and activations
- **Communication pattern**: Hierarchical concatenation reduces cross-device synchronization

## Experimental Validations
- Mixed precision (FP16) ensures GPU throughput saturation
- Large batch size (1024) prevents hardware idling
- Performance gains attributed to parallelization strategy improvements, not hardware underutilization

## Key Findings
1. Two-level partitioning outperforms traditional TP+PP baseline
2. Benefits consistent across both dense and MoE architectures
3. Greater improvements observed for MoE models (35.3% vs 31.7%)
4. Reduced communication overhead validates efficient partitioning strategy