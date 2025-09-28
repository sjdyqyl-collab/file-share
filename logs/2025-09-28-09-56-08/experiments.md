# XAttention: Experiments

## 1. Experimental Setup

### Models Evaluated
- **Natural Language**: Llama-3.1-8B-Instruct
- **Video Understanding**: Qwen2-VL-7B-Instruct  
- **Video Generation**: HunyuanVideo (Diffusion Transformer)

### Baselines
- **Dense Attention**: FlashAttention (FlashInfer framework)
- **Sparse Methods**: 
  - MInference (Vertical-Slash pattern)
  - FlexPrefill (γ=0.95, τ=0.1)
  - SeerAttention (with pretraining)

### Datasets
- **Language**: RULER (synthetic), LongBench (real-world)
- **Video Understanding**: VideoMME (900 videos, 254 hours)
- **Video Generation**: VBench (946 GPT-augmented prompts)

### XAttention Configurations
- **Stride S**: 8, 16 (primary), 4, 64 (ablation)
- **Threshold τ**: 0.9, 0.95, dynamic (0.8 average)
- **Block Size**: 8×8, 16×16
- **Warmup**: 5 steps for video generation

## 2. Language Understanding Results

### RULER Benchmark (Synthetic)

| Method | 4k | 8k | 16k | 32k | 64k | 128k | Avg |
|--------|-----|-----|------|------|------|-------|------|
| Full | 96.74 | 94.03 | 92.02 | 84.17 | 81.32 | 76.89 | 87.52 |
| FlexPrefill | 95.99 | 93.67 | 92.73 | 88.14 | 81.14 | 74.67 | 87.72 |
| MInference | 96.54 | 94.06 | 91.37 | 85.79 | 83.03 | 54.12 | 84.15 |
| SeerAttn | 84.43 | 79.55 | 79.80 | 72.95 | 64.79 | 51.61 | 72.18 |
| XAttn S=8 | 96.83 | 94.07 | 93.17 | 90.75 | 84.08 | 72.31 | 88.47 |
| XAttn S=16 | 96.11 | 93.95 | 93.56 | 90.64 | 83.12 | 71.11 | 88.08 |

**Key Findings**:
- XAttention (S=8) achieves highest average score (88.47)
- Outperforms full attention at several sequence lengths
- Maintains performance up to 128k tokens
- MInference and SeerAttention degrade significantly at long contexts

### LongBench Results (Real-world)

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Code | Avg |
|--------|---------------|--------------|---------------|----------|------|-----|
| Full | 31.44 | 25.07 | 29.40 | 16.89 | 17.00 | 40.34 |
| MInference | 31.59 | 24.82 | 29.53 | 17.03 | 16.46 | 40.30 |
| FlexPrefill | 27.30 | 28.56 | 27.66 | 17.20 | 15.14 | 36.83 |
| XAttention | 28.99 | 26.14 | 29.92 | 17.40 | 16.70 | 40.60 |

**Key Findings**:
- XAttention achieves highest average score (40.60)
- Performance close to full attention on individual tasks
- Demonstrates effectiveness in practical scenarios

## 3. Video Understanding Results

### VideoMME Benchmark

| Method | Short (%) | Medium (%) | Long (%) | Overall (%) |
|--------|-----------|------------|----------|-------------|
| Full | 72.1/78.1 | 63.9/69.4 | 55.1/60.2 | 63.7/69.2 |
| MInference | 71.7/77.6 | 62.3/67.9 | 55.2/59.8 | 63.1/68.4 |
| FlexPrefill | 71.4/77.4 | 62.6/68.3 | 53.8/57.3 | 62.6/67.7 |
| XAttention | 71.9/78.8 | 62.6/68.5 | 55.7/60.3 | 63.3/69.1 |

**Key Findings**:
- Best average performance among sparse methods
- Outperforms full attention on long video tasks
- Effective for videos up to 1 hour (1 fps)

## 4. Video Generation Results

### VBench Evaluation

| Configuration | PSNR (↑) | SSIM (↑) | LPIPS (↓) | Density (%, ↓) |
|---------------|----------|----------|-----------|---------------|
| Full (baseline) | - | - | - | 100.0 |
| XAttn τ=0.90 | 21.5 | 0.767 | 0.215 | 34.4 |
| XAttn τ=0.95 | 23.5 | 0.822 | 0.155 | 45.5 |

**Key Findings**:
- High fidelity with PSNR up to 23.5
- Over 50% sparsity achieved
- 5-step warmup preserves layout quality
- Visual similarity difficult to distinguish from full attention

## 5. Efficiency Analysis

### Attention Acceleration

| Context Length | 8k | 16k | 32k | 64k | 128k | 256k |
|----------------|-----|------|------|------|-------|-------|
| MInference | 7.6× | 6.0× | 3.9× | 2.2× | 1.0× | 0.2× |
| FlexPrefill | 4.2× | 2.5× | 1.1× | 0.8× | 0.4× | 0.2× |
| XAttn S=8 | 13.5× | 11.7× | 8.4× | 5.1× | 2.5× | 1.5× |
| XAttn S=16 | 9.8× | 7.1× | 4.3× | 3.2× | 2.2× | 1.7× |

**Key Findings**:
- Maximum 13.5× speedup at 256k tokens
- Consistent advantages across all lengths
- Other methods lose advantage at long contexts

### Pattern Selection Time

| Method | Pattern Search (ms) | Attention (ms) | Total (ms) |
|--------|-------------------|---------------|------------|
| Full | - | 89.6 | 89.6 |
| MInference | 73.8 | 15.8 | 89.6 |
| FlexPrefill | 20.8 | 9.3 | 30.1 |
| XAttn S=8 | 3.6 | 13.9 | 17.5 |
| XAttn S=16 | 3.6 | 14.3 | 17.9 |

**Key Findings**:
- 24.9× faster pattern selection than MInference
- 5.9× faster than FlexPrefill
- Lower attention density leads to faster sparse computation

### Density Analysis

| Context Length | S=4 | S=8 | S=16 |
|----------------|-----|------|-------|
| 4k | 51.73% | 52.16% | 55.38% |
| 8k | 40.96% | 43.77% | 43.55% |
| 16k | 27.43% | 27.49% | 28.91% |
| 32k | 21.09% | 20.97% | 27.93% |
| 64k | 9.43% | 10.98% | 11.32% |
| 128k | 6.20% | 6.89% | 7.32% |

**Key Findings**:
- Higher sparsity with longer contexts
- S=8 achieves optimal balance
- Up to 93.8% sparsity at 128k tokens

## 6. Ablation Studies

### Pattern Comparison (32k context)

| Pattern | S=8 Avg | S=8 Density | S=16 Avg | S=16 Density |
|---------|---------|-------------|----------|--------------|
| Random | 82.53 | 27.57% | 82.35 | 31.36% |
| Diagonal | 76.47 | 24.47% | 58.26 | 25.31% |
| Antidiagonal | 90.75 | 20.97% | 90.64 | 27.93% |

**Key Findings**:
- Antidiagonal pattern achieves highest accuracy
- Maintains lowest density across configurations
- Superior to random and diagonal patterns

### Stride Impact

| Stride | Average Score | Density |
|--------|---------------|---------|
| S=4 | 88.89 | 21.09% |
| S=8 | 88.47 | 20.97% |
| S=16 | 88.08 | 27.93% |
| S=64 | 81.21 | 39.88% |

**Key Findings**:
- S=64 too sparse, loses pattern detection capability
- S=8 provides optimal balance

### Selection Strategy Comparison

| Method | S=8 Avg | S=8 Density | S=16 Avg | S=16 Density |
|--------|---------|-------------|----------|--------------|
| Top-K | 84.13 | 19.92% | 83.11 | 30.15% |
| Top-Ratio | 85.42 | 21.00% | 84.24 | 27.00% |
| Threshold | 88.47 | 20.97% | 88.08 | 27.93% |

**Key Findings**:
- Threshold selection achieves optimal balance
- Top-K and Top-Ratio struggle with dynamic lengths

### Dynamic Threshold Impact

| Configuration | S=4 Avg | S=4 Density | S=8 Avg | S=8 Density |
|---------------|---------|-------------|---------|-------------|
| Fixed τ=0.9 | 87.51 | 23.06% | 84.96 | 26.13% |
| Dynamic | 88.89 | 21.09% | 88.47 | 20.97% |

**Key Findings**:
- Dynamic threshold improves both accuracy and sparsity
- Average threshold reduces from 0.9 to 0.8
- Better adaptation to different attention heads

## 7. Implementation Details

### Hardware Setup
- NVIDIA GPUs for acceleration
- FlashInfer framework for baseline comparisons
- Optimized sparse attention kernels

### Software Configuration
- Stride S=8, 16 for primary experiments
- Dynamic threshold prediction with M=1000
- Block size B=8 for most experiments
- Warmup strategy for video generation (5 steps)

### Reproducibility
- Public implementations of baselines used
- Official configurations for MInference and FlexPrefill
- Consistent random seeds across experiments
- Multiple runs averaged for stability