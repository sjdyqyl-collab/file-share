# XAttention: Experimental Evaluation

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

### Configuration Parameters
- XAttention: Stride S=8,16 with dynamic threshold prediction
- MInference: Vertical-Slash sparsity pattern (official config)
- FlexPrefill: γ=0.95, τ=0.1 (optimal from paper)
- SeerAttention: With pretrained Gare weights

## 3.2 Accuracy Results

### RULER Benchmark (Table 1)
| Method | 4k | 8k | 16k | 32k | 64k | 128k | Avg |
|--------|-----|-----|------|------|------|-------|------|
| Full | 96.74 | 94.03 | 92.02 | 84.17 | 81.32 | 76.89 | 87.52 |
| FlexPrefill | 95.99 | 93.67 | 92.73 | 88.14 | 81.14 | 74.67 | 87.72 |
| MInference | 96.54 | 94.06 | 91.37 | 85.79 | 83.03 | 54.12 | 84.15 |
| SeerAttn | 84.43 | 79.55 | 79.80 | 72.95 | 64.79 | 51.61 | 72.18 |
| XAttn S=8 | 96.83 | 94.07 | 93.17 | 90.75 | 84.08 | 72.31 | 88.47 |
| XAttn S=16 | 96.11 | 93.95 | 93.56 | 90.64 | 83.12 | 71.11 | 88.08 |

**Key Findings**:
- XAttention outperforms all sparse baselines
- Maintains accuracy comparable to full attention
- Robust performance across all sequence lengths
- S=8 configuration slightly better than S=16

### LongBench Results (Table 2)
XAttention achieves highest average score (40.60) across all real-world tasks:
- Single-Doc QA, Multi-Doc QA, Summarization
- Few-shot Learning, Code generation
- Performance remains close to full attention (40.34)

### Video Understanding (Table 3)
| Method | Short | Medium | Long | Overall |
|--------|--------|----------|--------|----------|
| Full | 72.1/78.1 | 63.9/69.4 | 55.1/60.2 | 63.7/69.2 |
| XAttention | 71.9/78.8 | 62.6/68.5 | 55.7/60.3 | 63.3/69.1 |

- XAttention outperforms full attention on long videos
- Best average performance among sparse methods
- Maintains quality across different video lengths

### Video Generation Results (Table 4)
| Threshold | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Density ↓ |
|-----------|---------|---------|----------|------------|
| τ=0.90 | 21.5 | 0.767 | 0.215 | 34.4% |
| τ=0.95 | 23.5 | 0.822 | 0.155 | 45.5% |

- High fidelity to full attention baseline
- PSNR up to 23.5, SSIM up to 0.822
- Achieves >50% sparsity with warmup strategy

## 3.3 Efficiency Results

### Attention Acceleration (Figure 4)
- **256k tokens**: Up to 13.5× speedup (S=8), 9.8× (S=16)
- **128k tokens**: 8.4× (S=8), 7.1× (S=16)
- **64k tokens**: 5.1× (S=8), 4.3× (S=16)
- Consistent speedup across all context lengths

### Pattern Selection Time (Figure 5)
- XAttention: 3.6ms pattern selection time
- MInference: 89.6ms (24.9× slower)
- FlexPrefill: 20.8ms (5.9× slower)

### Density Analysis (Table 5)
| SeqLen | S=4 | S=8 | S=16 |
|--------|------|------|-------|
| 4k | 51.73% | 52.16% | 55.38% |
| 8k | 40.96% | 43.77% | 43.55% |
| 16k | 27.43% | 27.49% | 28.91% |
| 32k | 21.09% | 20.97% | 27.93% |
| 64k | 9.43% | 10.98% | 11.32% |
| 128k | 6.20% | 6.89% | 7.32% |

- Higher sparsity (lower density) with longer sequences
- S=8 achieves best sparsity-accuracy tradeoff

## 3.4 Ablation Studies

### Pattern Comparison (Table 6)
| Pattern | 32k Avg | Density |
|---------|----------|----------|
| Random | 82.48 | 27.57% |
| Diagonal | 81.06 | 24.47% |
| Antidiagonal | 88.47 | 20.97% |

- Antidiagonal achieves highest accuracy with lowest density

### Stride Analysis (Table 7)
| Stride | Avg Score | Density |
|--------|-----------|----------|
| S=4 | 88.89 | 21.09% |
| S=8 | 88.47 | 20.97% |
| S=16 | 88.08 | 27.93% |
| S=64 | 81.21 | 39.88% |

- S=4,8,16 maintain good accuracy
- S=64 degrades performance significantly

### Selection Strategy (Table 8)
| Method | S=8 Avg | Density |
|--------|----------|----------|
| Top-K | 84.13 | 19.92% |
| Top-Ratio | 85.42 | 21.00% |
| Threshold (Ours) | 88.47 | 20.97% |

- Threshold-based selection achieves optimal balance

### Dynamic Threshold (Table 9)
| Configuration | S=8 Avg | Density |
|---------------|----------|----------|
| Fixed τ=0.9 | 84.96 | 26.13% |
| Dynamic τ | 88.47 | 20.97% |

- Dynamic threshold improves both accuracy and sparsity

## Key Experimental Insights

1. **Superior Performance**: XAttention consistently outperforms existing sparse attention methods
2. **Scalability**: Effective across 4k-256k token sequences
3. **Efficiency**: Pattern selection 24.9× faster than MInference
4. **Robustness**: Maintains accuracy across diverse tasks and domains
5. **Practicality**: Plug-and-play deployment without retraining