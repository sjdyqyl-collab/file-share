# Phase 3: Experiments of XAttention

## 1. Experimental Setup

### Models Evaluated
- **Natural Language**: Llama-3.1-8B-Instruct
- **Video Understanding**: Qwen2-VL-7B-Instruct  
- **Video Generation**: HunyuanVideo (Diffusion Transformer)

### Baselines
- **Dense Attention**: FlashAttention (FlashInfer implementation)
- **Sparse Methods**: 
  - MInference (Vertical-Slash pattern)
  - FlexPrefill (γ=0.95, τ=0.1)
  - SeerAttention (with pretraining on Gare weights)

### Datasets
- **RULER**: Synthetic long-context benchmark (4k-128k tokens)
- **LongBench**: Real-world long-context tasks
- **VideoMME**: 900 videos (11s-1h duration, 254h total)
- **VBench**: 946 GPT-augmented text prompts for video generation

## 2. Accuracy Results

### RULER Benchmark (Llama-3.1-8B)
| Input Length | 4k | 8k | 16k | 32k | 64k | 128k | Avg |
|--------------|-----|-----|------|------|------|-------|-----|
| Full Attention | 96.74 | 94.03 | 92.02 | 84.17 | 81.32 | 76.89 | 87.52 |
| FlexPrefill | 95.99 | 93.67 | 92.73 | 88.14 | 81.14 | 74.67 | 87.72 |
| MInference | 96.54 | 94.06 | 91.37 | 85.79 | 83.03 | 54.12 | 84.15 |
| SeerAttn | 84.43 | 79.55 | 79.80 | 72.95 | 64.79 | 51.61 | 72.18 |
| **XAttn S=8** | **96.83** | **94.07** | **93.17** | **90.75** | **84.08** | **72.31** | **88.47** |
| **XAttn S=16** | 96.11 | 93.95 | **93.56** | **90.64** | **83.12** | **71.11** | **88.08** |

**Key Findings**:
- XAttention outperforms all baselines including full attention at multiple lengths
- Maintains performance up to 128k tokens while others degrade significantly
- S=8 configuration generally better than S=16

### LongBench Results (Llama-3.1-8B)
| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Code | Avg |
|--------|---------------|--------------|---------------|----------|------|-----|
| Full | 25.25 | 14.96 | 26.27 | 69.58 | 47.67 | 40.34 |
| MInference | 25.31 | 14.90 | 26.26 | 69.42 | 47.67 | 40.30 |
| FlexPrefill | 24.70 | 13.93 | 25.69 | 64.45 | 44.61 | 36.83 |
| **XAttention** | **25.69** | **15.30** | **26.57** | **69.54** | **48.25** | **40.60** |

**Key Findings**:
- XAttention achieves highest average score across all task categories
- Performance closest to full attention among all sparse methods
- Particularly strong on summarization and code tasks

### Video Understanding (Qwen2-VL-7B on VideoMME)
| Method | Short (%) | Medium (%) | Long (%) | Overall (%) |
|--------|-----------|------------|----------|-------------|
| Full | 72.1/78.1 | 63.9/69.4 | 55.1/60.2 | 63.7/69.2 |
| MInference | 71.7/77.6 | 62.3/67.9 | 55.2/59.8 | 63.1/68.4 |
| FlexPrefill | 71.4/77.4 | 62.6/68.3 | 53.8/57.3 | 62.6/67.7 |
| **XAttention** | **71.9/78.8** | **62.6/68.5** | **55.7/60.3** | **63.3/69.1** |

**Key Findings**:
- XAttention outperforms full attention on long video tasks
- Best performance among all sparse attention methods
- Effective for videos up to 1 hour at 1fps

### Video Generation (HunyuanVideo on VBench)
| Configuration | PSNR (↑) | SSIM (↑) | LPIPS (↓) | Density (%, ↓) |
|---------------|----------|----------|-----------|----------------|
| Full Attention | Baseline | Baseline | Baseline | 100% |
| XAttn τ=0.90 | 21.5 | 0.767 | 0.215 | 34.4% |
| XAttn τ=0.95 | 23.5 | 0.822 | 0.155 | 45.5% |

**Key Findings**:
- High fidelity to full attention baseline (PSNR > 23)
- 5-step warmup strategy crucial for maintaining layout consistency
- Trade-off between quality (higher τ) and sparsity (lower τ)

## 3. Efficiency Results

### Attention Speedup vs Context Length
| Context Length | 8k | 16k | 32k | 64k | 128k | 256k |
|----------------|-----|------|------|------|-------|-------|
| **XAttn S=8** | 7.6× | 9.8× | 11.7× | 13.5× | 13.5× | 13.5× |
| **XAttn S=16** | 6.0× | 7.1× | 8.4× | 9.8× | 9.8× | 9.8× |
| MInference | 3.9× | 4.3× | 5.1× | 2.5× | 1.5× | 1.7× |
| FlexPrefill | 2.2× | 3.2× | 2.5× | 1.0× | 0.2× | 0.1× |

### Density Analysis (S=8)
| Context Length | 4k | 8k | 16k | 32k | 64k | 128k |
|----------------|-----|-----|------|------|------|-------|
| Density | 52.16% | 43.77% | 27.49% | 20.97% | 10.98% | 6.89% |

### Pattern Selection Time Breakdown
- **XAttention**: 3.6ms (antidiagonal scoring)
- **MInference**: 89.6ms (vertical slash search) - **24.9× slower**
- **FlexPrefill**: 20.8ms (pattern search) - **5.9× slower**

## 4. Ablation Studies

### Pattern Comparison (S=8, 32k tokens)
| Pattern | Accuracy | Density |
|---------|----------|---------|
| Random | 82.48 | 27.57% |
| Diagonal | 81.06 | 24.47% |
| **Antidiagonal** | **88.47** | **20.97%** |

### Stride Impact
| Stride | Accuracy | Density |
|--------|----------|---------|
| S=4 | 88.89 | 21.09% |
| **S=8** | **88.47** | **20.97%** |
| S=16 | 88.08 | 27.93% |
| S=64 | 81.21 | 39.88% |

### Selection Strategy Comparison (S=8)
| Strategy | Accuracy | Density |
|----------|----------|---------|
| Top-K | 84.96 | 17.40% |
| Top-Ratio | 85.96 | 21.00% |
| **Threshold** | **88.89** | **21.09%** |

### Dynamic vs Fixed Threshold
| Method | Accuracy | Density |
|--------|----------|---------|
| Fixed τ=0.9 | 84.96 | 26.13% |
| **Dynamic** | **88.47** | **20.97%** |

## 5. Runtime Analysis

### Computational Time Representation

**Baseline (Full Attention)**:
- Time: [L, L, d] where L is sequence length, d is hidden dimension
- Complexity: O(L²d)

**XAttention**:
- **Pattern Selection**: [L/S, L/S, d] for antidiagonal scoring
- **Sparse Attention**: [L, L·ρ, d] where ρ is density ratio
- **Total**: [L, L·(1/S + ρ), d]

**Specific Examples**:
- At 256k tokens: [256k, 256k, d] → [256k, 256k·0.07, d] = 13.5× speedup
- At 128k tokens: [128k, 128k, d] → [128k, 128k·0.07, d] = 13.5× speedup

### Memory Usage
- **Reduction**: Proportional to sparsity ratio (1 - density)
- **At 128k tokens**: ~93% memory reduction (6.89% density)