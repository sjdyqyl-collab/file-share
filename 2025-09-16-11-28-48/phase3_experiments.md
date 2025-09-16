# Phase 3: Experiments and Results

## 1. Experimental Setup

### Models Tested
- **HunyuanVideo**: 13B parameters, 720p 8-second videos, 50 denoising steps
- **CogVideoX1.5-5B**: 5B parameters, 720p 10-second videos, 50 denoising steps

### Baselines
1. **Full Attention**: Original dense attention mechanism
2. **MInference**: Dynamic pattern with offline search + online approximation
3. **Sparse VideoGen**: Static pattern designed for DiTs
4. **AdaSpa Variants**:
   - AdaSpa (w/o head adaptive): Uniform sparsity across heads
   - AdaSpa (w/o LSE cache): Without LSE caching optimization

### Dataset
- **VBench**: Default dataset for testing
- **CogVideoX1.5-5B**: VBench with prompt optimization following CogVideoX guidelines

### Metrics
- **Quality**: VBench Score (%), PSNR (↑), SSIM (↑), LPIPS (↓)
- **Efficiency**: Latency (seconds), Speedup (×)
- **Hardware**: Single A100 GPU-80GB

## 2. Main Results

### HunyuanVideo Results
| Method | VBench (%) ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Latency (s) | Speedup |
|--------|---------------|---------|---------|----------|-------------|---------|
| Full Attention | 80.10 | - | - | - | 3213.76 | 1.00× |
| MInference | 79.17 | 22.53 | 0.7435 | 0.3550 | 2532.80 | 1.27× |
| Sparse VideoGen | 79.39 | 27.61 | 0.8683 | 0.1703 | 2035.59 | 1.58× |
| AdaSpa (w/o head adaptive) | 79.64 | 28.51 | 0.8825 | 0.1574 | 1823.34 | 1.76× |
| AdaSpa (w/o LSE cache) | 80.16 | 28.97 | 0.8898 | 0.1481 | 1877.13 | 1.71× |
| **AdaSpa (ours)** | **80.13** | **29.07** | **0.8905** | **0.1478** | **1810.23** | **1.78×** |

### CogVideoX1.5-5B Results
| Method | VBench (%) ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Latency (s) | Speedup |
|--------|---------------|---------|---------|----------|-------------|---------|
| Full Attention | 81.16 | - | - | - | 3135.24 | 1.00× |
| MInference | 65.30 | 10.31 | 0.3113 | 0.6820 | 2258.35 | 1.39× |
| Sparse VideoGen | 79.40 | 18.98 | 0.6465 | 0.3632 | 2061.42 | 1.52× |
| AdaSpa (w/o head adaptive) | 81.54 | 22.99 | 0.8133 | 0.2203 | 1915.88 | 1.64× |
| AdaSpa (w/o LSE cache) | 81.73 | 23.14 | 0.8255 | 0.2091 | 1961.71 | 1.60× |
| **AdaSpa (ours)** | **81.90** | **23.25** | **0.8267** | **0.2067** | **1888.14** | **1.66×** |

## 3. Key Findings

### Performance Analysis
- **AdaSpa consistently outperforms** all baselines in both quality and efficiency
- **MInference struggles** with approximate search leading to lowest accuracy
- **Sparse VideoGen** performs well but lacks dynamic adaptation
- **Head-adaptive mechanism** proves crucial (79.64→80.13 VBench on HunyuanVideo)
- **LSE caching** provides additional speedup (1.71×→1.78× on HunyuanVideo)

### Quality Trends
- **PSNR**: AdaSpa achieves 29.07 vs 22.53 (MInference) and 27.61 (Sparse VideoGen)
- **SSIM**: AdaSpa reaches 0.8905 vs 0.7435 (MInference) and 0.8683 (Sparse VideoGen)
- **LPIPS**: AdaSpa maintains 0.1478 vs 0.3550 (MInference) and 0.1703 (Sparse VideoGen)

## 4. Ablation Studies

### Quality-Sparsity Trade-off
- **AdaSpa maintains highest quality** across all sparsity levels (0.7-0.9)
- **Competitors show significant degradation** as sparsity increases
- **Linear degradation** for AdaSpa vs abrupt decline for others

### Warmup Impact
- **All methods benefit from warmup** (2-10 steps)
- **AdaSpa consistently best** across all warmup configurations
- **Quality plateau** reached after 5 warmup steps
- **Similarity improves** with increased warmup

### Search Strategy Evaluation
| Search Strategy (Ts) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|----------------------|---------|---------|----------|
| {10} | 28.96 | 0.8879 | 0.1509 |
| **{10, 30}** | **29.07** | **0.8905** | **0.1478** |
| {10, 20, 30} | 28.93 | 0.8894 | 0.1500 |
| {10, 20, 30, 40} | 28.93 | 0.8898 | 0.1494 |

- **Optimal strategy**: Ts = {10, 30} balances quality and efficiency
- **Diminishing returns** beyond 2 search points
- **Quality degradation** with excessive searches

## 5. Scaling Study

### Video Length vs Speedup
- **Linear scaling**: Speedup increases with video length
- **24-second videos**: Achieves 4.01× speedup
- **Excellent scalability**: Maintains performance gains across lengths

### Performance Scaling
- **2-second**: 2.01× speedup
- **8-second**: 2.79× speedup  
- **16-second**: 3.65× speedup
- **24-second**: 4.01× speedup

## 6. Runtime Analysis

### Search Overhead
- **Online search time**: <5% of full attention generation time
- **Fused LSE-Cached**: Reduces search passes from 2 to 1
- **Head-adaptive**: Minimal additional computation

### Memory Efficiency
- **Block-wise processing**: O(Lb) memory vs O(L²) for dense
- **LSE caching**: Reuses intermediate results across steps
- **Sparse computation**: Only computes selected blocks

## 7. Qualitative Results

### Visual Comparison
- **Figure 1**: AdaSpa maintains visual fidelity closest to original videos
- **Artifact reduction**: Fewer visual artifacts compared to MInference and Sparse VideoGen
- **Temporal consistency**: Better frame-to-frame continuity

### Perceptual Quality
- **VBench scores**: Consistently highest across both models
- **Human evaluation**: Subjective quality matches quantitative metrics
- **Prompt adherence**: Maintains text-video alignment better than baselines