# Phase 3: Experiments Extraction - AdaSpa

## 5.1 Main Results

### Experimental Setup
**Models Tested:**
- HunyuanVideo (13B parameters)
- CogVideoX1.5-5B

**Video Specifications:**
- HunyuanVideo: 720p, 8-second videos
- CogVideoX1.5-5B: 720p, 10-second videos
- Both: 50 denoising steps

**Baselines:**
- Sparse VideoGen (static pattern)
- MInference (dynamic pattern)
- AdaSpa variants:
  - AdaSpa (w/o head adaptive)
  - AdaSpa (w/o lse cache)

**Dataset:**
- VBench dataset (default)
- CogVideoX1.5-5B: VBench with prompt optimization

### Performance Results

#### HunyuanVideo Results
| Method | VBench (%) ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Latency (s) | Speedup |
|--------|--------------|--------|--------|---------|-------------|---------|
| Full Attention | 80.10 | - | - | - | 3213.76 | 1.00× |
| MInference | 79.17 | 22.53 | 0.7435 | 0.3550 | 2532.80 | 1.27× |
| Sparse VideoGen | 79.39 | 27.61 | 0.8683 | 0.1703 | 2035.59 | 1.58× |
| AdaSpa (w/o head adaptive) | 79.64 | 28.51 | 0.8825 | 0.1574 | 1823.34 | 1.76× |
| AdaSpa (w/o lse cache) | 80.16 | 28.97 | 0.8898 | 0.1481 | 1877.13 | 1.71× |
| **AdaSpa (ours)** | **80.13** | **29.07** | **0.8905** | **0.1478** | **1810.23** | **1.78×** |

#### CogVideoX1.5-5B Results
| Method | VBench (%) ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Latency (s) | Speedup |
|--------|--------------|--------|--------|---------|-------------|---------|
| Full Attention | 81.16 | - | - | - | 3135.24 | 1.00× |
| MInference | 65.30 | 10.31 | 0.3113 | 0.6820 | 2258.35 | 1.39× |
| Sparse VideoGen | 79.40 | 18.98 | 0.6465 | 0.3632 | 2061.42 | 1.52× |
| AdaSpa (w/o head adaptive) | 81.54 | 22.99 | 0.8133 | 0.2203 | 1915.88 | 1.64× |
| AdaSpa (w/o lse cache) | 81.73 | 23.14 | 0.8255 | 0.2091 | 1961.71 | 1.60× |
| **AdaSpa (ours)** | **81.90** | **23.25** | **0.8267** | **0.2067** | **1888.14** | **1.66×** |

## 5.2 Ablation Studies

### Quality-Sparsity Trade-off
- **AdaSpa** maintains highest video quality across all sparsity levels (0.7-0.9)
- **Competitors** show significant quality drop as sparsity increases
- **Linear degradation** for AdaSpa vs abrupt decline for others

### Warmup Impact
- **All methods** show decreased similarity with reduced warmup
- **AdaSpa** maintains best performance across all warmup setups
- **Video quality** remains almost unchanged with warmup increase
- **Warmup primarily affects** similarity, not generation quality

### Search Strategy Evaluation
| Search Strategy (Ts) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---------------------|--------|--------|---------|
| {10} | 28.9629 | 0.8879 | 0.1509 |
| {10, 30} | 29.0749 | 0.8905 | 0.1478 |
| {10, 20, 30} | 28.9343 | 0.8894 | 0.1500 |
| {10, 20, 30, 40} | 28.9313 | 0.8898 | 0.1494 |

**Finding:** Increasing searches beyond {10,30} provides limited benefit and may decrease quality.

## 5.3 Scaling Study

### Video Length vs Speedup
- **Configuration**: sparsity=0.9, block_size=64, Ts={0,30}
- **Scaling Results**:
  - 8s video: ~1.8× speedup
  - 16s video: ~2.8× speedup  
  - 24s video: **4.01× speedup**

**Conclusion:** AdaSpa demonstrates excellent scalability with increasing video length.

## Key Experimental Findings

### Performance Superiority
1. **Best Quality**: Highest VBench, PSNR, SSIM scores across both models
2. **Best Speed**: Highest speedup (1.78× HunyuanVideo, 1.66× CogVideoX1.5-5B)
3. **Robust Performance**: Maintains quality across sparsity levels

### Component Effectiveness
- **Head-adaptive mechanism**: Improves quality metrics (compare AdaSpa vs w/o head adaptive)
- **LSE caching**: Improves speed (compare AdaSpa vs w/o lse cache)
- **Block pattern**: More effective than continuous patterns for DiT sparsity

### Practical Advantages
- **Plug-and-play**: Single line code change integration
- **No training required**: Training-free and data-free
- **Scalable**: Speedup increases with video length
- **Orthogonal**: Compatible with other acceleration techniques