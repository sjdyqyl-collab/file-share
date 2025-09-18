# Phase 3: Experiments and Results - XAttention

## 3.1 Experimental Setup

### Models Evaluated
- **Natural Language**: Llama-3.1-8B-Instruct
- **Video Understanding**: Qwen2-VL-7B-Instruct  
- **Video Generation**: HunyuanVideo (DiT architecture)

### Baselines
- **Dense Attention**: FlashAttention (FlashInfer implementation)
- **Sparse Methods**: MInference, FlexPrefill, SeerAttention

### Datasets
- **RULER**: Synthetic long-context benchmark (4k-128k tokens)
- **LongBench**: Real-world long-context tasks
- **VideoMME**: 900 videos (11s-1h duration) for video understanding
- **VBench**: 946 prompts for video generation evaluation

## 3.2 Accuracy Results

### RULER Benchmark (Table 1)
| Input Length | 4k | 8k | 16k | 32k | 64k | 128k | Avg |
|--------------|-----|-----|------|------|------|-------|-----|
| Full Attention | 96.74 | 94.03 | 92.02 | 84.17 | 81.32 | 76.89 | 87.52 |
| FlexPrefill | 95.99 | 93.67 | 92.73 | 88.14 | 81.14 | 74.67 | 87.72 |
| MInference | 96.54 | 94.06 | 91.37 | 85.79 | 83.03 | 54.12 | 84.15 |
| SeerAttn | 84.43 | 79.55 | 79.80 | 72.95 | 64.79 | 51.61 | 72.18 |
| **XAttn S=8** | **96.83** | **94.07** | **93.17** | **90.75** | **84.08** | **72.31** | **88.47** |
| **XAttn S=16** | **96.11** | **93.95** | **93.56** | **90.64** | **83.12** | **71.11** | **88.08** |

**Key Findings**: XAttention surpasses all baselines including full attention at several sequence lengths.

### LongBench Results (Table 2)
| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Code | Average |
|--------|---------------|--------------|---------------|----------|------|---------|
| Full | 49.14 | 40.34 | - | - | - | 40.34 |
| MInference | 49.93 | 40.30 | - | - | - | 40.30 |
| FlexPrefill | 47.54 | 36.83 | - | - | - | 36.83 |
| **XAttention** | **50.84** | **40.60** | - | - | - | **40.60** |

**Key Findings**: XAttention achieves highest average score across all real-world tasks.

### Video Understanding (Table 3)
| Method | Short (%) | Medium (%) | Long (%) | Overall (%) |
|--------|-----------|------------|----------|-------------|
| Full | 72.1/78.1 | 63.9/69.4 | 55.1/60.2 | 63.7/69.2 |
| MInference | 71.7/77.6 | 62.3/67.9 | 55.2/59.8 | 63.1/68.4 |
| FlexPrefill | 71.4/77.4 | 62.6/68.3 | 53.8/57.3 | 62.6/67.7 |
| **XAttention** | **71.9/78.8** | **62.6/68.5** | **55.7/60.3** | **63.3/69.1** |

**Key Findings**: XAttention outperforms full attention on long videos (1 hour duration).

### Video Generation (Table 4)
| XAttn τ | PSNR (↑) | SSIM (↑) | LPIPS (↓) | Density (%, ↓) |
|---------|----------|----------|-----------|----------------|
| 0.90 | 21.5 | 0.767 | 0.215 | 34.4 |
| **0.95** | **23.5** | **0.822** | **0.155** | 45.5 |

**Key Findings**: High fidelity to full attention baseline with 50%+ sparsity.

## 3.3 Efficiency Results

### Attention Acceleration (Figure 4)
- **Maximum speedup**: 13.5× at 256k tokens (S=16)
- **Secondary speedup**: 9.8× at 256k tokens (S=8)
- **Consistent advantage**: Maintains speedup across all context lengths

### Time Breakdown (Figure 5)
- **Pattern selection speedup**:
  - 24.9× faster than MInference
  - 5.9× faster than FlexPrefill
- **Sparse attention computation**: Substantial speedup due to lower density

### Density Analysis (Table 5)
| SeqLen | Stride 4 | Stride 8 | Stride 16 |
|--------|----------|----------|-----------|
| 4k | 51.73% | 52.16% | 55.38% |
| 8k | 40.96% | 43.77% | 43.55% |
| 16k | 27.43% | 27.49% | 28.91% |
| 32k | 21.09% | 20.97% | 27.93% |
| 64k | 9.43% | 10.98% | 11.32% |
| 128k | 6.20% | 6.89% | 7.32% |

**Key Findings**: Density decreases with sequence length, achieving 6-7% at 128k tokens.

## 3.4 Ablation Studies

### Antidiagonal Pattern Effectiveness (Table 6)
| Pattern | 32k Avg | Density | 32k Avg | Density |
|---------|---------|---------|---------|---------|
| Random | 82.48 | 27.57% | 80.94 | 31.36% |
| Diagonal | 81.06 | 24.47% | 79.63 | 25.31% |
| **Antidiagonal** | **88.47** | **20.97%** | **88.08** | **27.93%** |

**Key Findings**: Antidiagonal pattern achieves highest accuracy with lowest density.

### Stride Size Impact (Table 7)
| Stride | Average | Density |
|--------|---------|---------|
| 4 | 88.89 | 21.09% |
| 8 | 88.47 | 20.97% |
| 16 | 88.08 | 27.93% |
| 64 | 81.21 | 39.88% |

**Key Findings**: S=8 provides best balance; S=64 causes significant accuracy drop.

### Selection Strategy Comparison (Table 8)
| Strategy | S=4 Avg | Density | S=8 Avg | Density | S=16 Avg | Density |
|----------|---------|---------|---------|---------|----------|---------|
| Top-K | 84.96 | 17.40% | 84.13 | 19.92% | 83.11 | 30.15% |
| Ratio | 85.96 | 21.00% | 85.42 | 21.00% | 84.24 | 27.00% |
| **Threshold** | **88.89** | **21.09%** | **88.47** | **20.97%** | **88.08** | **27.93%** |

**Key Findings**: Threshold-based selection outperforms Top-K and Ratio methods.

### Dynamic Threshold vs Fixed (Table 9)
| Method | S=4 Avg | Density | S=8 Avg | Density | S=16 Avg | Density |
|--------|---------|---------|---------|---------|----------|---------|
| τ=0.9 | 87.51 | 23.06% | 84.96 | 26.13% | 85.83 | 28.36% |
| **Dynamic τ** | **88.89** | **21.09%** | **88.47** | **20.97%** | **88.08** | **27.93%** |

**Key Findings**: Dynamic threshold prediction improves both accuracy and sparsity.

## Runtime Analysis

### Baseline Full Attention
- **Computation**: [L, d] × [d, L] → [L, L] = O(L²d)
- **Example**: At 256k tokens, [256k, 4096] × [4096, 256k] → [256k, 256k]

### XAttention
- **Pattern Selection**: [L/S, d] × [d, L/S] → [L/S, L/S] = O((L/S)²d)
- **Sparse Attention**: ρ × [L, d] × [d, L] → [L, L] = O(ρL²d)
- **Total**: O((L/S)²d + ρL²d)
- **Example**: At 256k tokens with S=16, ρ=7.32%:
  - Pattern selection: [16k, 4096] × [4096, 16k] → [16k, 16k]
  - Sparse attention: 7.32% × [256k, 4096] × [4096, 256k] → [256k, 256k]

### Communication Overhead
- **None**: XAttention is a single-device method, no communication required