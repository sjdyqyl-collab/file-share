# Phase 3: Experiments and Results

## 5.1 Main Results

### Experimental Setup
- **Models**: HunyuanVideo (13B) and CogVideoX1.5-5B
- **Resolution**: 720p videos
- **Duration**: 8 seconds (HunyuanVideo), 10 seconds (CogVideoX1.5-5B)
- **Steps**: 50 denoising steps
- **Hardware**: Single A100 GPU-80GB
- **Dataset**: VBench (with prompt optimization for CogVideoX1.5-5B)

### Baseline Comparison

#### HunyuanVideo Results
| Method | VBench (↑) | PSNR (↑) | SSIM (↑) | LPIPS (↓) | Latency (s) | Speedup |
|--------|------------|----------|----------|-----------|-------------|---------|
| Full Attention | 80.10 | - | - | - | 3213.76 | 1.00× |
| MInference | 79.17 | 22.53 | 0.7435 | 0.3550 | 2532.80 | 1.27× |
| Sparse VideoGen | 79.39 | 27.61 | 0.8683 | 0.1703 | 2035.59 | 1.58× |
| AdaSpa (ours) | 80.13 | 29.07 | 0.8905 | 0.1478 | 1810.23 | 1.78× |

#### CogVideoX1.5-5B Results
| Method | VBench (↑) | PSNR (↑) | SSIM (↑) | LPIPS (↓) | Latency (s) | Speedup |
|--------|------------|----------|----------|-----------|-------------|---------|
| Full Attention | 81.16 | - | - | - | 3135.24 | 1.00× |
| MInference | 65.30 | 10.31 | 0.3113 | 0.6820 | 2258.35 | 1.39× |
| Sparse VideoGen | 79.40 | 18.98 | 0.6465 | 0.3632 | 2061.42 | 1.52× |
| AdaSpa (ours) | 81.90 | 23.25 | 0.8267 | 0.2067 | 1888.14 | 1.66× |

### Ablation Study Results

#### AdaSpa Variants Performance
| Model | Variant | VBench (↑) | PSNR (↑) | Speedup |
|-------|---------|------------|----------|---------|
| HunyuanVideo | w/o head adaptive | 79.64 | 28.51 | 1.76× |
| HunyuanVideo | w/o LSE cache | 80.16 | 28.97 | 1.71× |
| CogVideoX1.5 | w/o head adaptive | 81.54 | 22.99 | 1.64× |
| CogVideoX1.5 | w/o LSE cache | 81.73 | 23.14 | 1.60× |

## 5.2 Ablation Studies

### Quality-Sparsity Trade-off
- **AdaSpa** maintains highest video quality across all sparsity levels
- **VBench scores**: Linear decrease vs abrupt drop in baselines
- **PSNR/SSIM**: Gradual degradation vs sharp decline in MInference
- **LPIPS**: Consistent improvement over baselines at all sparsity levels

### Warmup Impact
- **All methods** show decreased similarity with fewer warmup steps
- **AdaSpa** consistently outperforms baselines across all warmup setups
- **Video quality** remains stable regardless of warmup duration
- **Optimal warmup**: 10 steps (default configuration)

### Search Strategy Analysis
| Search Strategy (Ts) | PSNR (↑) | SSIM (↑) | LPIPS (↓) |
|---------------------|----------|----------|-----------|
| {10} | 28.9629 | 0.8879 | 0.1509 |
| {10, 30} | 29.0749 | 0.8905 | 0.1478 |
| {10, 20, 30} | 28.9343 | 0.8894 | 0.1500 |
| {10, 20, 30, 40} | 28.9313 | 0.8898 | 0.1494 |

- **Optimal**: {10, 30} strategy balances quality and efficiency
- **Diminishing returns**: Additional searches beyond threshold decrease quality

## 5.3 Scaling Study

### Video Length Scaling Results
| Video Length (s) | Speedup |
|------------------|---------|
| 8 | 1.78× |
| 16 | 2.79× |
| 24 | 4.01× |

- **Linear scaling**: Speedup increases with video length
- **Efficiency gains**: More pronounced for longer sequences
- **Memory efficiency**: Block-wise computation scales better

## Runtime Analysis

### Computational Complexity Comparison
```
Full Attention: Get_Time[L, d, L] = O(L²d)
MInference: Get_Time[0.3L, d, L] + Get_Time[L, d, 0.3L] + Search_Cost
Sparse VideoGen: Get_Time[0.25L, d, L] + Get_Time[L, d, 0.25L]
AdaSpa: Get_Time[0.2L, d, L] + Get_Time[L, d, 0.2L] + Search_Cost[0.05×Full]
```

### Memory Usage
- **Full Attention**: O(L²) attention matrix storage
- **AdaSpa**: O(L·B) block-wise storage (B=64)
- **Cache overhead**: O(L) for LSE storage across steps

## Quality Metrics Details

### Perceptual Similarity Metrics
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (higher is better)  
- **LPIPS**: Learned Perceptual Image Patch Similarity (lower is better)

### Video Quality Metrics
- **VBench**: Comprehensive benchmark considering both pixel accuracy and perceptual consistency
- **Frame consistency**: Maintained through row-wise uniformity constraint
- **Text alignment**: Enhanced via text sink mechanism

## Implementation Details

### Configuration Parameters
- **Sparsity**: 0.8 (80% sparsity)
- **Block size**: 64 tokens
- **Search intervals**: Ts = {10, 30} steps
- **Warmup steps**: 10
- **Head adaptation**: Recall threshold = 0.8

### Performance Characteristics
- **Search overhead**: <5% of total generation time
- **Memory reduction**: ~80% compared to full attention
- **Quality loss**: <1% VBench score degradation
- **Integration**: Single-line code change for existing DiTs