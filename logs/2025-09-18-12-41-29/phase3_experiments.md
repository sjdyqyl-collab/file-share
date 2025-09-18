# Phase 3: Experiments and Results - XAttention

## Experimental Setup

### Models Evaluated
- **Natural Language**: Llama-3.1-8B-Instruct
- **Video Understanding**: Qwen2-VL-7B-Instruct  
- **Video Generation**: HunyuanVideo (Diffusion Transformer)

### Baselines Compared
- **Full Attention**: FlashAttention via FlashInfer
- **Sparse Methods**: MInference, FlexPrefill, SeerAttention
- **Configuration**: Strict adherence to public implementations

### Datasets
- **RULER**: Synthetic long-context benchmark (4K-128K tokens)
- **LongBench**: Real-world long-context tasks
- **VideoMME**: 900 videos (11s-1h duration) for video understanding
- **VBench**: 946 prompts for video generation evaluation

## Accuracy Results

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
- XAttention S=8 achieves highest average score (88.47)
- Outperforms full attention at multiple sequence lengths
- Maintains performance up to 128K tokens while others degrade

### LongBench Results (Table 2)
- **XAttention Average**: 40.60
- **Full Attention**: 40.34  
- **MInference**: 40.30
- **FlexPrefill**: 36.83

**Individual Task Performance**: Close to full attention across all categories (Single-Doc QA, Multi-Doc QA, Summarization, Few-shot Learning, Code)

### Video Understanding (Table 3)
| Method | Short (%) | Medium (%) | Long (%) | Overall (%) |
|--------|-----------|------------|----------|-------------|
| Full | 72.1/78.1 | 63.9/69.4 | 55.1/60.2 | 63.7/69.2 |
| MInference | 71.7/77.6 | 62.3/67.9 | 55.2/59.8 | 63.1/68.4 |
| FlexPrefill | 71.4/77.4 | 62.6/68.3 | 53.8/57.3 | 62.6/67.7 |
| XAttention | 71.9/78.8 | 62.6/68.5 | 55.7/60.3 | 63.3/69.1 |

**Key Achievement**: Outperforms full attention on long video tasks

### Video Generation (Table 4)
| Configuration | PSNR (↑) | SSIM (↑) | LPIPS (↓) | Density (%) |
|---------------|----------|----------|-----------|-------------|
| τ=0.90 | 21.5 | 0.767 | 0.215 | 34.4 |
| τ=0.95 | 23.5 | 0.822 | 0.155 | 45.5 |

**Quality Metrics**: Both configurations achieve high fidelity with PSNR > 21.5, SSIM > 0.76, LPIPS < 0.22

## Efficiency Results

### Attention Acceleration (Figure 4)
- **Maximum Speedup**: 13.5× at 256K tokens (S=8)
- **Secondary Speedup**: 9.8× at 256K tokens (S=16)
- **Consistent Performance**: Maintains speedup across all context lengths

### Pattern Selection Time (Figure 5)
- **vs. MInference**: 24.9× faster pattern selection
- **vs. FlexPrefill**: 5.9× faster pattern selection
- **Breakdown**: Pattern search time significantly reduced while maintaining lower attention density

### Density Analysis (Table 5)
| SeqLen | Stride 4 | Stride 8 | Stride 16 |
|--------|----------|----------|-----------|
| 4k | 51.73% | 52.16% | 55.38% |
| 8k | 40.96% | 43.77% | 43.55% |
| 16k | 27.43% | 27.49% | 28.91% |
| 32k | 21.09% | 20.97% | 27.93% |
| 64k | 9.43% | 10.98% | 11.32% |
| 128k | 6.20% | 6.89% | 7.32% |

**Trend**: Density decreases with increasing sequence length, achieving >90% sparsity at 128K tokens

## Ablation Studies

### Pattern Comparison (Table 6)
| Pattern | 32k Avg | Density | Performance |
|---------|---------|---------|-------------|
| Random | 82.53 | 27.57% | Baseline |
| Diagonal | 76.47 | 24.47% | Lower accuracy |
| Antidiagonal | 90.75 | 20.97% | **Best performance** |

### Stride Impact (Table 7)
| Stride | Average Score | Density |
|--------|---------------|---------|
| S=4 | 88.89 | 21.09% |
| S=8 | 88.47 | 20.97% |
| S=16 | 88.08 | 27.93% |
| S=64 | 81.21 | 39.88% |

**Finding**: S=64 too sparse, fails to distinguish slash patterns effectively

### Selection Strategy (Table 8)
| Method | S=8 Avg | Density | Adaptability |
|--------|---------|---------|--------------|
| Top-K | 84.13 | 19.92% | Fixed computation |
| Top-Ratio | 85.42 | 21.00% | Fixed ratio |
| Threshold (XAttention) | 88.47 | 20.97% | **Dynamic, optimal** |

### Threshold Prediction (Table 9)
| Configuration | S=8 Avg | Density | Improvement |
|---------------|---------|---------|-------------|
| Fixed τ=0.9 | 84.96 | 26.13% | Baseline |
| Dynamic Threshold | 88.47 | 20.97% | **+3.51 accuracy, -5.16% density** |

## Runtime Analysis

### Matrix Multiplication Representation
- **Baseline Full Attention**: [L, d, L] → L²d operations
- **XAttention Sparse**: [L, d, L×density] → L²d×density operations
- **Pattern Selection**: [L/S, L/S, B²/S²] for antidiagonal scoring

### Communication Costs
- **No additional communication** for single-device inference
- **Multi-device**: Standard attention communication patterns apply
- **Memory**: Reduced from O(L²) to O(L²×density)

### Example Runtime Breakdown (256K tokens)
- **Full Attention**: [256K, 128, 256K] → 8.6T operations
- **XAttention (7% density)**: [256K, 128, 18K] → 600B operations
- **Speedup**: ~14.3× theoretical, 13.5× achieved (accounting for overhead)

## Experimental Limitations

### Evaluation Scope
- **Limited Model Sizes**: Primarily 8B parameter models
- **Domain Coverage**: Language, video understanding/generation only
- **Hardware**: Specific GPU configurations (not fully specified)

### Measurement Considerations
- **End-to-end vs. Attention-only**: Results focus on attention acceleration
- **Batch Effects**: Single-batch evaluation primarily
- **Statistical Significance**: Limited discussion of variance across runs

### Reproducibility Factors
- **Implementation Details**: Some hyperparameters not exhaustively explored
- **Threshold Selection**: Dynamic programming method complex
- **Hardware Dependencies**: Results may vary across different GPU architectures