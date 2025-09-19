# Phase 3: Experiments and Results - XAttention

## 3.1 Experimental Setup

### Models Evaluated
- **Natural Language**: Llama-3.1-8B-Instruct
- **Video Understanding**: Qwen2-VL-7B-Instruct  
- **Video Generation**: HunyuanVideo (Diffusion Transformer)

### Baselines Compared
- **Dense Attention**: FlashAttention (FlashInfer implementation)
- **Sparse Methods**: MInference, FlexPrefill, SeerAttention

### Datasets
- **RULER**: Synthetic long-context benchmark (4K-128K tokens)
- **LongBench**: Real-world long-context tasks
- **Video-MME**: 900 videos (11s-1h duration) for video understanding
- **VBench**: 946 prompts for video generation evaluation

## 3.2 Accuracy Results

### RULER Benchmark (Table 1)
| Input Length | 4k | 8k | 16k | 32k | 64k | 128k | Avg |
|--------------|-----|-----|------|------|------|-------|-----|
| Full Attention | 96.74 | 94.03 | 92.02 | 84.17 | 81.32 | 76.89 | 87.52 |
| FlexPrefill | 95.99 | 93.67 | 92.73 | 88.14 | 81.14 | 74.67 | 87.72 |
| MInference | 96.54 | 94.06 | 91.37 | 85.79 | 83.03 | 54.12 | 84.15 |
| **XAttention S=8** | **96.83** | **94.07** | **93.17** | **90.75** | **84.08** | **72.31** | **88.47** |
| **XAttention S=16** | **96.11** | **93.95** | **93.56** | **90.64** | **83.12** | **71.11** | **88.08** |

**Key Finding**: XAttention outperforms all baselines including full attention at multiple sequence lengths.

### LongBench Results (Table 2)
- **XAttention Average Score**: 40.60 (highest among all methods)
- **Performance**: Close to full attention (40.34) while significantly better than sparse baselines
- **Individual Tasks**: Maintains accuracy across diverse task types

### Video Understanding (Table 3)
| Method | Short (%) | Medium (%) | Long (%) | Overall (%) |
|--------|-----------|------------|----------|-------------|
| Full Attention | 72.1/78.1 | 63.9/69.4 | 55.1/60.2 | 63.7/69.2 |
| **XAttention** | **71.9/78.8** | **62.6/68.5** | **55.7/60.3** | **63.3/69.1** |

**Achievement**: Outperforms full attention on long video tasks.

### Video Generation Results (Table 4)
| Configuration | PSNR (↑) | SSIM (↑) | LPIPS (↓) | Density (%) |
|---------------|----------|----------|-----------|-------------|
| τ=0.90 | 21.5 | 0.767 | 0.215 | 34.4 |
| **τ=0.95** | **23.5** | **0.822** | **0.155** | **45.5** |

**Qualitative**: With 5-step warmup, achieves visual fidelity indistinguishable from full attention.

## 3.3 Efficiency Results

### Attention Acceleration (Figure 4)
- **Maximum Speedup**: 13.5× at 256K tokens (S=16, ρ=7.32%)
- **Second Best**: 9.8× at 256K tokens (S=8, ρ=6.89%)
- **Trend**: Consistent speedup across all context lengths (8K-256K)

### Density Analysis (Table 5)
| SeqLen | Stride 4 | Stride 8 | Stride 16 |
|--------|----------|----------|-----------|
| 4k | 51.73% | 52.16% | 55.38% |
| 8k | 40.96% | 43.77% | 43.55% |
| 16k | 27.43% | 27.49% | 28.91% |
| 32k | 21.09% | 20.97% | 27.93% |
| 64k | 9.43% | 10.98% | 11.32% |
| 128k | 6.20% | 6.89% | 7.32% |

**Observation**: Sparsity increases (density decreases) with longer sequences.

### Time Breakdown (Figure 5)
- **Pattern Selection**: 
  - XAttention: 14.3ms (S=8), 15.8ms (S=16)
  - MInference: 73.8ms (24.9× slower)
  - FlexPrefill: 89.6ms (5.9× slower)
- **Sparse Attention**: 
  - XAttention: 3.6ms (S=8), 9.3ms (S=16)
  - Benefits from lower density selection

## 3.4 Ablation Studies

### Pattern Comparison (Table 6)
| Pattern | 32k Accuracy | Density |
|---------|-------------|---------|
| Random | 82.53 | 27.57% |
| Diagonal | 76.47 | 24.47% |
| **Antidiagonal** | **90.75** | **20.97%** |

**Conclusion**: Antidiagonal pattern achieves best accuracy with lowest density.

### Stride Size Impact (Table 7)
| Stride | Average Score | Density |
|--------|---------------|---------|
| S=4 | 88.89 | 21.09% |
| **S=8** | **88.47** | **20.97%** |
| **S=16** | **88.08** | **27.93%** |
| S=64 | 81.21 | 39.88% |

**Finding**: S=8 and S=16 provide optimal balance.

### Selection Strategy (Table 8)
| Method | S=8 Accuracy | Density |
|--------|--------------|---------|
| Top-K | 84.96 | 17.40% |
| Ratio | 85.96 | 21.00% |
| **Threshold** | **88.89** | **21.09%** |

**Advantage**: Threshold-based selection handles dynamic lengths better.

### Dynamic vs Fixed Threshold (Table 9)
| Configuration | S=8 Accuracy | Density |
|---------------|--------------|---------|
| Fixed τ=0.9 | 84.96 | 26.13% |
| **Dynamic τ** | **88.47** | **20.97%** |

**Improvement**: Dynamic threshold reduces density by 5.16% while improving accuracy.

## Runtime Summary
- **Full Attention**: [L, d, L] - O(L²d)
- **XAttention**: [L, d, L·ρ] + [L, d, L/S] - O(L²ρd + L²d/S)
- **Example**: L=256K, d=4096, ρ=6.89%, S=8 → 13.5× speedup
- **Communication**: None (training-free method)