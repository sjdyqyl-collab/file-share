# Phase 3: Experiments Extraction - AdaSpa Paper

## Experimental Setup

### Models Tested
1. **HunyuanVideo (13B)**
   - Resolution: 720p
   - Duration: 8 seconds
   - Steps: 50 denoising steps
   - Sequence length: ~110K tokens

2. **CogVideoX1.5-5B**
   - Resolution: 720p  
   - Duration: 10 seconds
   - Steps: 50 denoising steps
   - Prompt optimization applied

### Baselines
1. **Sparse VideoGen** - Static pattern approach
2. **MInference** - Dynamic pattern with offline search + online approximation
3. **AdaSpa Variants**:
   - AdaSpa (w/o head adaptive) - Uniform sparsity across heads
   - AdaSpa (w/o LSE cache) - Without LSE caching optimization

### Dataset
- **VBench** - Standard video generation benchmark
- CogVideoX1.5-5B uses prompt-optimized VBench dataset

### Metrics
- **Quality**: VBench Score, PSNR, SSIM, LPIPS
- **Efficiency**: Latency (seconds), Speedup (×)
- **Hardware**: Single A100 GPU-80GB

## Main Results

### HunyuanVideo Results
| Method | VBench (↑) | PSNR (↑) | SSIM (↑) | LPIPS (↓) | Latency (s) | Speedup |
|--------|------------|----------|----------|-----------|-------------|---------|
| Full Attention | 80.10 | - | - | - | 3213.76 | 1.00× |
| MInference | 79.17 | 22.53 | 0.7435 | 0.3550 | 2532.80 | 1.27× |
| Sparse VideoGen | 79.39 | 27.61 | 0.8683 | 0.1703 | 2035.59 | 1.58× |
| AdaSpa (w/o head) | 79.64 | 28.51 | 0.8825 | 0.1574 | 1823.34 | 1.76× |
| AdaSpa (w/o cache) | 80.16 | 28.97 | 0.8898 | 0.1481 | 1877.13 | 1.71× |
| **AdaSpa (ours)** | **80.13** | **29.07** | **0.8905** | **0.1478** | **1810.23** | **1.78×** |

### CogVideoX1.5-5B Results
| Method | VBench (↑) | PSNR (↑) | SSIM (↑) | LPIPS (↓) | Latency (s) | Speedup |
|--------|------------|----------|----------|-----------|-------------|---------|
| Full Attention | 81.16 | - | - | - | 3135.24 | 1.00× |
| MInference | 65.30 | 10.31 | 0.3113 | 0.6820 | 2258.35 | 1.39× |
| Sparse VideoGen | 79.40 | 18.98 | 0.6465 | 0.3632 | 2061.42 | 1.52× |
| AdaSpa (w/o head) | 81.54 | 22.99 | 0.8133 | 0.2203 | 1915.88 | 1.64× |
| AdaSpa (w/o cache) | 81.73 | 23.14 | 0.8255 | 0.2091 | 1961.71 | 1.60× |
| **AdaSpa (ours)** | **81.90** | **23.25** | **0.8267** | **0.2067** | **1888.14** | **1.66×** |

## Key Findings

### Performance Analysis
1. **AdaSpa consistently achieves best quality and efficiency**
   - Highest VBench scores across both models
   - Highest speedups: 1.78× (HunyuanVideo), 1.66× (CogVideoX1.5-5B)
   - Maintains quality metrics close to full attention

2. **Baselines underperform**
   - MInference: Lowest quality due to approximate search limitations
   - Sparse VideoGen: Good but static patterns limit adaptability

3. **Ablation study validates design choices**
   - Head-adaptive mechanism: +0.5 VBench, +0.5 PSNR improvement
   - LSE caching: +0.07× speedup improvement

## Ablation Studies

### Quality-Sparsity Trade-off
- **Sparsity Range**: 0.7, 0.8, 0.9
- **Observation**: AdaSpa maintains highest quality across all sparsity levels
- **Trend**: Linear quality degradation vs abrupt drops in baselines

### Warmup Impact
- **Tested**: 0, 2, 5, 10 warmup steps
- **Finding**: Quality increases with warmup, AdaSpa maintains best performance
- **Conclusion**: 10-step warmup optimal for stability

### Search Strategy Evaluation
| Search Strategy (Ts) | PSNR (↑) | SSIM (↑) | LPIPS (↓) |
|---------------------|----------|----------|-----------|
| {10} | 28.96 | 0.8879 | 0.1509 |
| {10, 30} | **29.07** | **0.8905** | **0.1478** |
| {10, 20, 30} | 28.93 | 0.8894 | 0.1500 |
| {10, 20, 30, 40} | 28.93 | 0.8898 | 0.1494 |

**Optimal**: {10, 30} strategy balances quality and computational overhead

## Scaling Study

### Video Length Scaling
- **Tested**: 2s, 4s, 8s, 16s, 24s videos
- **Result**: Speedup increases with video length
- **Maximum**: 4.01× speedup for 24-second videos
- **Trend**: Linear scaling relationship

### Computational Scaling
- **Memory**: O(L²) → O((1-sparsity)L²) reduction
- **Time**: Consistent speedup across different sequence lengths
- **Search Overhead**: Amortized across long sequences

## Visualization Results
(Figure 1 in paper)
- **Qualitative comparison**: AdaSpa maintains visual fidelity
- **Artifacts**: Minimal compared to baselines
- **Consistency**: Preserves temporal coherence

## Statistical Significance
- **Confidence**: Results averaged over multiple seeds
- **Stability**: Low variance across runs
- **Reproducibility**: Open-source implementation available