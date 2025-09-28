# XAttention: Experimental Results and Analysis

## 1. Experimental Setup

### Models Evaluated
- **Llama-3.1-8B-Instruct**: Natural language processing tasks
- **Qwen2-VL-7B-Instruct**: Video understanding tasks  
- **HunyuanVideo**: Video generation tasks (Diffusion Transformer)

### Baselines Compared
- **FlashAttention**: Dense attention baseline (FlashInfer implementation)
- **MInference**: Vertical-Slash sparsity pattern
- **FlexPrefill**: Context-aware sparse attention (γ=0.95, τ=0.1)
- **SeerAttention**: Pretrained gate parameters

### Datasets
- **RULER**: Synthetic long-context benchmark (4K-128K tokens)
- **LongBench**: Real-world long-context tasks
- **VideoMME**: 900 videos (11s-1h duration) for video understanding
- **VBench**: 946 prompts for video generation evaluation

## 2. Accuracy Results

### RULER Benchmark (Llama-3.1-8B)
| Input Length | 4k | 8k | 16k | 32k | 64k | 128k | Avg |
|--------------|-----|-----|------|------|------|-------|------|
| Full Attention | 96.74 | 94.03 | 92.02 | 84.17 | 81.32 | 76.89 | 87.52 |
| FlexPrefill | 95.99 | 93.67 | 92.73 | 88.14 | 81.14 | 74.67 | 87.72 |
| MInference | 96.54 | 94.06 | 91.37 | 85.79 | 83.03 | 54.12 | 84.15 |
| XAttention S=8 | 96.83 | 94.07 | 93.17 | 90.75 | 84.08 | 72.31 | 88.47 |
| XAttention S=16 | 96.11 | 93.95 | 93.56 | 90.64 | 83.12 | 71.11 | 88.08 |

**Key Findings**:
- XAttention outperforms all sparse baselines
- Maintains performance closer to full attention even at 128K tokens
- S=8 slightly better than S=16 for accuracy

### LongBench Results (Llama-3.1-8B)
| Method | Avg Score |
|--------|-----------|
| Full Attention | 40.34 |
| MInference | 40.30 |
| FlexPrefill | 36.83 |
| XAttention | 40.60 |

- XAttention achieves highest average score among all methods
- Individual task performance remains close to full attention

### Video Understanding (QwenVL-2-7B on VideoMME)
| Video Length | Short | Medium | Long | Overall |
|--------------|--------|---------|------|---------|
| Full | 63.7% | 69.2% | 60.2% | 69.2% |
| MInference | 63.1% | 68.4% | 59.8% | 68.4% |
| FlexPrefill | 62.6% | 67.7% | 57.3% | 67.7% |
| XAttention | 63.3% | 69.1% | 60.3% | 69.1% |

- XAttention outperforms full attention on long videos
- Best performance among sparse attention methods

### Video Generation (HunyuanVideo on VBench)
| Configuration | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Density ↓ |
|---------------|--------|--------|---------|-----------|
| Full Attention | - | - | - | 100% |
| XAttention τ=0.90 | 21.5 | 0.767 | 0.215 | 34.4% |
| XAttention τ=0.95 | 23.5 | 0.822 | 0.155 | 45.5% |

- High fidelity to full attention baseline
- 5-step warmup prevents layout shifts
- Trade-off between quality (τ=0.95) and sparsity (τ=0.90)

## 3. Efficiency Results

### Attention Speedup vs Context Length
| Context Length | 8K | 16K | 32K | 64K | 128K | 256K |
|----------------|-----|------|------|------|-------|-------|
| XAttention S=8 | 7.6× | 6.0× | 3.9× | 2.2× | 1.0× | 0.2× |
| XAttention S=16 | 9.8× | 7.1× | 4.3× | 3.2× | 2.2× | 1.7× |
| MInference | 4.2× | 2.5× | 1.1× | 0.8× | 0.4× | 0.2× |
| FlexPrefill | 3.1× | 2.4× | 1.0× | 0.8× | 0.6× | 0.1× |

**Peak Performance**: 13.5× speedup at 256K tokens (S=16)

### Pattern Selection Time Breakdown
- **XAttention**: 14.3ms (antidiagonal scoring)
- **MInference**: 356.7ms (vertical/slash search)
- **FlexPrefill**: 84.2ms (context-aware selection)

**Speedup**: 24.9× faster than MInference, 5.9× faster than FlexPrefill

### Density Analysis
| Context Length | S=4 | S=8 | S=16 |
|----------------|------|------|-------|
| 4K | 51.73% | 52.16% | 55.38% |
| 8K | 40.96% | 43.77% | 43.55% |
| 16K | 27.43% | 27.49% | 28.91% |
| 32K | 21.09% | 20.97% | 27.93% |
| 64K | 9.43% | 10.98% | 11.32% |
| 128K | 6.20% | 6.89% | 7.32% |

- Higher sparsity (lower density) with longer contexts
- S=8 achieves best sparsity-accuracy trade-off

## 4. Ablation Studies

### Pattern Comparison (32K context)
| Pattern | Accuracy | Density |
|---------|----------|---------|
| Random | 82.48 | 27.57% |
| Diagonal | 81.06 | 24.47% |
| Antidiagonal | 88.47 | 20.97% |

**Conclusion**: Antidiagonal pattern superior in both accuracy and sparsity

### Stride Size Impact
| Stride | Avg Accuracy | Density |
|--------|--------------|---------|
| S=4 | 88.89 | 21.09% |
| S=8 | 88.47 | 20.97% |
| S=16 | 88.08 | 27.93% |
| S=64 | 81.21 | 39.88% |

**Finding**: S=8 optimal balance; S=64 too coarse for pattern detection

### Selection Strategy Comparison
| Method | Avg Accuracy | Density |
|--------|--------------|---------|
| Top-K | 84.13 | 19.92% |
| Top-Ratio | 85.42 | 21.00% |
| Threshold (Dynamic) | 88.47 | 20.97% |

**Advantage**: Dynamic threshold handles variable sequence lengths better

### Minimum Threshold Prediction
| Configuration | Avg Accuracy | Density |
|---------------|--------------|---------|
| Fixed τ=0.9 | 84.96 | 26.13% |
| Dynamic τ | 88.47 | 20.97% |

**Benefit**: 3.5% accuracy improvement + 5% additional sparsity

## 5. Runtime Analysis

### Baseline Full Attention
- **Computation**: [L, L, d] → O(L²d)
- **Example**: 256K tokens, d=4096 → [256000, 256000, 4096]

### XAttention Sparse Attention
- **Pattern Selection**: [B, B, d] per block → O(L·B·d/S)
- **Sparse Computation**: [L·τ, L·τ, d] → O(L²d·τ²)
- **Example**: 256K tokens, τ=0.073, B=64, S=8
  - Pattern: [64, 64, 4096] × (256000/64) = 4096 blocks
  - Sparse: [18688, 18688, 4096] (7.32% density)

### Communication Overhead
- **None**: Pure computation optimization, no additional communication
- **Memory**: Reduced from O(L²) to O(L²·τ²)

## 6. Reproducibility Details

### Key Parameters
- **Block Size**: B=64 (typical)
- **Stride**: S=8 (optimal) or S=16 (faster)
- **Threshold**: τ=0.9 (default), optimized via DP
- **Warmup**: 5 steps for video generation

### Hardware Configuration
- **GPU**: NVIDIA DGX systems
- **Framework**: FlashInfer integration
- **Precision**: FP16/BF16 standard

### Reproduction Commands
```bash
# Language tasks
python evaluate.py --model llama-3.1-8b --dataset ruler --stride 8 --threshold 0.9

# Video understanding  
python evaluate.py --model qwen2-vl-7b --dataset videomme --stride 16 --threshold 0.9

# Video generation
python generate.py --model hunyuanvideo --dataset vbench --stride 8 --threshold 0.95 --warmup 5
```