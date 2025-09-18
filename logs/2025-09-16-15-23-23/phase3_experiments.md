# Phase 3: Experiments and Results

## 5.1 Experimental Setup

### Models Tested
1. **HunyuanVideo (13B)**: 720p, 8-second videos, 50 denoising steps
2. **CogVideoX1.5-5B**: 720p, 10-second videos, 50 denoising steps

### Baselines
- **Sparse VideoGen**: Static pattern method
- **MInference**: Dynamic pattern with approximate search
- **AdaSpa Variants**:
  - AdaSpa (w/o head adaptive): Uniform sparsity across heads
  - AdaSpa (w/o LSE cache): Without LSE caching optimization

### Evaluation Metrics
- **Quality**: VBench Score (%), PSNR, SSIM, LPIPS
- **Efficiency**: Latency (seconds), Speedup (×)
- **Hardware**: Single A100 GPU-80GB

## 5.2 Main Results

### HunyuanVideo Results
| Method | VBench (%) ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Latency (s) | Speedup |
|--------|---------------|---------|---------|----------|-------------|---------|
| Full Attention | 80.10 | - | - | - | 3213.76 | 1.00× |
| MInference | 79.17 | 22.53 | 0.7435 | 0.3550 | 2532.80 | 1.27× |
| Sparse VideoGen | 79.39 | 27.61 | 0.8683 | 0.1703 | 2035.59 | 1.58× |
| AdaSpa (w/o head adaptive) | 79.64 | 28.51 | 0.8825 | 0.1574 | 1823.34 | 1.76× |
| AdaSpa (w/o LSE cache) | 80.16 | 28.97 | 0.8898 | 0.1481 | 1877.13 | 1.71× |
| **AdaSpa (ours)** | **80.13** | **29.07** | **0.8905** | **0.1478** | **1810.23** | **1.78×** |

### CogVideoX1.5 Results
| Method | VBench (%) ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Latency (s) | Speedup |
|--------|---------------|---------|---------|----------|-------------|---------|
| Full Attention | 81.16 | - | - | - | 3135.24 | 1.00× |
| MInference | 65.30 | 10.31 | 0.3113 | 0.6820 | 2258.35 | 1.39× |
| Sparse VideoGen | 79.40 | 18.98 | 0.6465 | 0.3632 | 2061.42 | 1.52× |
| AdaSpa (w/o head adaptive) | 81.54 | 22.99 | 0.8133 | 0.2203 | 1915.88 | 1.64× |
| AdaSpa (w/o LSE cache) | 81.73 | 23.14 | 0.8255 | 0.2091 | 1961.71 | 1.60× |
| **AdaSpa (ours)** | **81.90** | **23.25** | **0.8267** | **0.2067** | **1888.14** | **1.66×** |

## 5.3 Ablation Studies

### Quality-Sparsity Trade-off
- **AdaSpa maintains highest quality across all sparsity levels (0.7-0.9)**
- **Competitors show significant degradation as sparsity increases**
- **Linear degradation for AdaSpa vs. abrupt decline for others**

### Warmup Impact
- **All methods benefit from warmup (2-10 steps)**
- **AdaSpa consistently outperforms across all warmup configurations**
- **Quality plateau reached after ~5 warmup steps**

### Search Strategy Analysis
| Search Strategy (Ts) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|----------------------|---------|---------|----------|
| {10} | 28.96 | 0.8879 | 0.1509 |
| **{10, 30}** | **29.07** | **0.8905** | **0.1478** |
| {10, 20, 30} | 28.93 | 0.8894 | 0.1500 |
| {10, 20, 30, 40} | 28.93 | 0.8898 | 0.1494 |

**Finding**: Optimal at 2 searches (steps 10, 30); more searches don't improve quality

## 5.4 Scaling Study

### Video Length Scaling (sparsity=0.9, block_size=64, Ts={0,30})
- **Linear scaling benefits**: Speedup increases with video length
- **24-second videos**: Achieves 4.01× speedup
- **Demonstrates**: Excellent scalability for longer sequences

## 5.5 Key Performance Insights

### Efficiency Gains
1. **Best overall speedup**: 1.78× (HunyuanVideo), 1.66× (CogVideoX1.5)
2. **Search overhead**: <5% of full attention time
3. **Memory efficiency**: Block-wise processing reduces memory footprint

### Quality Preservation
1. **Negligible quality loss**: <1% VBench score drop
2. **Superior PSNR**: +1.46 over Sparse VideoGen (HunyuanVideo)
3. **Better perceptual quality**: Lower LPIPS scores across all tests

### Robustness
1. **Consistent performance**: Across different models and datasets
2. **Head-adaptive benefit**: +0.56 PSNR improvement over uniform sparsity
3. **LSE cache benefit**: +0.05 speedup improvement over non-cached version

## Runtime Analysis

### Complexity Comparison
- **Full Attention**: O(L²d) = [L, d, L] matrix multiplication
- **AdaSpa**: O((1-sparsity)L²d) = [(1-sparsity)L, d, L] matrix multiplication
- **Search Overhead**: O(L²d/B²) for block-level analysis

### Practical Runtime
- **HunyuanVideo**: 3213.76s → 1810.23s (1.78× speedup)
- **CogVideoX1.5**: 3135.24s → 1888.14s (1.66× speedup)
- **Search time**: ~90-150s per search (steps 10, 30)