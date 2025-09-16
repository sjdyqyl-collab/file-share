# Experimental Results - AdaSpa

## Experimental Setup

### Models Tested
- **HunyuanVideo**: 13B parameters, 720p 8-second videos, 50 steps
- **CogVideoX1.5-5B**: 720p 10-second videos, 50 steps

### Baselines
- **Sparse VideoGen**: Static pattern approach
- **MInference**: Dynamic pattern with offline search + online approximation
- **AdaSpa variants**:
  - w/o head adaptive: Uniform sparsity across heads
  - w/o LSE cache: No cached LSE optimization

### Metrics
- **Quality**: VBench Score, PSNR, SSIM, LPIPS
- **Efficiency**: Latency (seconds), Speedup (×)
- **Dataset**: VBench (with prompt optimization for CogVideoX1.5-5B)

## Main Results

### HunyuanVideo Performance
| Method | VBench (%) ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Latency (s) | Speedup |
|--------|---------------|---------|---------|----------|-------------|---------|
| Full Attention | 80.10 | - | - | - | 3213.76 | 1.00× |
| MInference | 79.17 | 22.53 | 0.7435 | 0.3550 | 2532.80 | 1.27× |
| Sparse VideoGen | 79.39 | 27.61 | 0.8683 | 0.1703 | 2035.59 | 1.58× |
| AdaSpa (w/o head) | 79.64 | 28.51 | 0.8825 | 0.1574 | 1823.34 | 1.76× |
| AdaSpa (w/o cache) | 80.16 | 28.97 | 0.8898 | 0.1481 | 1877.13 | 1.71× |
| **AdaSpa (ours)** | **80.13** | **29.07** | **0.8905** | **0.1478** | **1810.23** | **1.78×** |

### CogVideoX1.5-5B Performance
| Method | VBench (%) ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Latency (s) | Speedup |
|--------|---------------|---------|---------|----------|-------------|---------|
| Full Attention | 81.16 | - | - | - | 3135.24 | 1.00× |
| MInference | 65.30 | 10.31 | 0.3113 | 0.6820 | 2258.35 | 1.39× |
| Sparse VideoGen | 79.40 | 18.98 | 0.6465 | 0.3632 | 2061.42 | 1.52× |
| AdaSpa (w/o head) | 81.54 | 22.99 | 0.8133 | 0.2203 | 1915.88 | 1.64× |
| AdaSpa (w/o cache) | 81.73 | 23.14 | 0.8255 | 0.2091 | 1961.71 | 1.60× |
| **AdaSpa (ours)** | **81.90** | **23.25** | **0.8267** | **0.2067** | **1888.14** | **1.66×** |

## Ablation Studies

### Quality-Sparsity Trade-off
- **AdaSpa**: Maintains highest quality across all sparsity levels (0.7-0.9)
- **Competitors**: Significant quality drop as sparsity increases
- **Observation**: Linear quality decrease vs abrupt decline in other methods

### Warmup Impact
- **Setup**: Tested 0, 2, 5, 10 warmup steps
- **Finding**: 
  - All methods: Similarity decreases with fewer warmup steps
  - AdaSpa: Consistently best across all warmup configurations
  - Quality: Minimal impact from warmup duration

### Search Strategy Analysis
| Search Steps (Ts) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------------------|---------|---------|----------|
| {10} | 28.96 | 0.8879 | 0.1509 |
| {10, 30} | **29.07** | **0.8905** | **0.1478** |
| {10, 20, 30} | 28.93 | 0.8894 | 0.1500 |
| {10, 20, 30, 40} | 28.93 | 0.8898 | 0.1494 |

- **Optimal**: {10, 30} provides best balance
- **Observation**: More searches ≠ better quality due to pattern similarity

## Scaling Study
- **Configuration**: sparsity=0.9, block_size=64, Ts={0,30}
- **Results**: 
  - Speedup increases with video length
  - **4.01× speedup** achieved for 24-second videos
  - Demonstrates excellent scalability for long-form content

## Key Findings
1. **Superior Performance**: Consistently outperforms baselines in both quality and speed
2. **Robustness**: Maintains quality across varying sparsity levels
3. **Scalability**: Benefits increase with video length
4. **Efficiency**: Minimal overhead from online precise search (<5% of total time)
5. **Adaptability**: Head-adaptive strategy crucial for optimal performance