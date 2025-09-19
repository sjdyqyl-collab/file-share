# Phase 3: Experimental Details and Results

## 3.1 Experimental Setup

### Models Tested
1. **Language Model**: Llama-3.1-8B-Instruct
   - Tasks: RULER, LongBench
   - Config: Dynamic threshold prediction, S=8/16, M=1000

2. **Video Understanding**: Qwen2-VL-7B-Instruct
   - Task: VideoMME
   - Config: S=16, τ=0.9

3. **Video Generation**: HunyuanVideo (DiT architecture)
   - Task: VBench (946 prompts)
   - Config: S=8, τ=0.9/0.95, 5-step warmup

### Baselines Compared
- **Dense Attention**: FlashAttention (FlashInfer)
- **Sparse Methods**: 
  - MInference (Vertical-Slash pattern)
  - FlexPrefill (γ=0.95, τ=0.1)
  - SeerAttention (with Gare weights)

### Datasets
1. **RULER**: Synthetic long-context benchmark
   - Customizable sequence lengths (4k-128k)
   - Tasks: needle-in-haystack, multi-hop tracing

2. **LongBench**: Real-world long-context tasks
   - Single/Multi-doc QA, Summarization, Few-shot learning
   - 16 diverse tasks

3. **VideoMME**: 900 videos, 254 hours total
   - Duration: 11s to 1h
   - Categories: Short/Medium/Long videos

4. **VBench**: 946 GPT-augmented prompts
   - Resolution: 720×1280
   - 129 frames, 50 denoising steps

## 3.2 Accuracy Results

### RULER Benchmark (Table 1)
| Input Length | 4k | 8k | 16k | 32k | 64k | 128k | Avg |
|--------------|-----|-----|------|------|------|-------|------|
| Full | 96.74 | 94.03 | 92.02 | 84.17 | 81.32 | 76.89 | 87.52 |
| FlexPrefill | 95.99 | 93.67 | 92.73 | 88.14 | 81.14 | 74.67 | 87.72 |
| MInference | 96.54 | 94.06 | 91.37 | 85.79 | 83.03 | 54.12 | 84.15 |
| XAttn S=8 | 96.83 | 94.07 | 93.17 | 90.75 | 84.08 | 72.31 | 88.47 |
| XAttn S=16 | 96.11 | 93.95 | 93.56 | 90.64 | 83.12 | 71.11 | 88.08 |

**Key Finding**: XAttention outperforms all baselines including full attention at several lengths.

### LongBench Results (Table 2)
- **XAttention Average**: 40.60 (highest among sparse methods)
- **Full Attention**: 40.34
- **Performance**: Close to full attention across all 16 tasks

### Video Understanding (Table 3)
| Category | Short | Medium | Long | Overall |
|----------|--------|---------|------|---------|
| Full | 63.7% | 69.2% | - | 69.2% |
| XAttention | 63.3% | 69.1% | - | 69.1% |

**Key Finding**: Outperforms full attention on long videos.

### Video Generation (Table 4)
| τ | PSNR↑ | SSIM↑ | LPIPS↓ | Density↓ |
|---|--------|--------|---------|----------|
| 0.90 | 21.5 | 0.767 | 0.215 | 34.4% |
| 0.95 | 23.5 | 0.822 | 0.155 | 45.5% |

**Key Finding**: High fidelity (PSNR 23.5, SSIM 0.822) with 50%+ sparsity.

## 3.3 Efficiency Results

### Attention Speedup (Figure 4)
- **256k tokens**: 13.5× (S=8), 9.8× (S=16)
- **128k tokens**: 8.4× (S=8), 7.1× (S=16)
- **Pattern**: Consistent speedup across all lengths

### Time Breakdown (Figure 5)
- **Pattern Selection**: 24.9× faster than MInference, 5.9× faster than FlexPrefill
- **Attention Computation**: Reduced due to lower density

### Density Analysis (Table 5)
| SeqLen | S=8 | S=16 |
|--------|------|------|
| 4k | 52.16% | 55.38% |
| 8k | 43.77% | 43.55% |
| 16k | 27.49% | 28.91% |
| 32k | 20.97% | 27.93% |
| 64k | 10.98% | 11.32% |
| 128k | 6.89% | 7.32% |

**Pattern**: Higher sparsity with longer sequences.

## 3.4 Ablation Studies

### Antidiagonal Pattern Effectiveness (Table 6)
| Pattern | 32k Avg | Density |
|---------|---------|---------|
| Random | 82.35 | 31.36% |
| Diagonal | 58.26 | 25.31% |
| Antidiagonal | 90.64 | 27.93% |

**Conclusion**: Antidiagonal achieves best accuracy-density trade-off.

### Stride Sensitivity (Table 7)
| Stride | Avg Score | Density |
|--------|-----------|---------|
| S=4 | 88.89 | 21.09% |
| S=8 | 88.47 | 20.97% |
| S=16 | 88.08 | 27.93% |
| S=64 | 81.21 | 39.88% |

**Conclusion**: S=64 too aggressive, S=4/8/16 good balance.

### Selection Strategy Comparison (Table 8)
| Method | Avg | Density |
|--------|-----|---------|
| Top-K | 84.13 | 19.92% |
| Ratio | 85.42 | 21.00% |
| Threshold | 88.47 | 20.97% |

**Conclusion**: Threshold-based selection best for dynamic lengths.

### Minimum Threshold Prediction (Table 9)
| Method | Avg | Density |
|--------|-----|---------|
| Fixed τ=0.9 | 84.96 | 26.13% |
| Dynamic τ | 88.47 | 20.97% |

**Conclusion**: Dynamic threshold improves both accuracy and sparsity.

## Runtime Analysis

### Baseline (Full Attention)
- **Computation**: [L, d, L] where L=sequence length, d=head dimension
- **Time**: O(L²d)

### XAttention
- **Pattern Selection**: [L/S, d, L/S] for antidiagonal scoring + O(L²/S) for selection
- **Sparse Attention**: [density×L, d, density×L] 
- **Total Time**: O(L²d×density + L²/S)

### Measured Speedups
- **256k tokens**: 13.5× speedup (S=8, density=6.89%)
- **128k tokens**: 9.8× speedup (S=16, density=7.32%)
- **Pattern Selection Overhead**: <5% of total time