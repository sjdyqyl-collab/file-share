# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Configuration
- **GPUs**: 16 × NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16)
- **Framework**: Compatible with model parallel frameworks

### Model Configurations

#### Dense Transformer Model
- **Layers**: 2-layer Dense Transformer
- **Attention heads**: 16 heads
- **Head dimension**: 512 per head
- **Hidden size**: 16 × 512 = 8192 (total embedding dimension)
- **MLP hidden size**: 32768

#### MoE Transformer Model  
- **Layers**: 2-layer Mixture-of-Experts Transformer
- **Experts per layer**: 4 experts
- **Attention heads**: 16 heads
- **Head dimension**: 512 per head
- **Hidden size**: 16 × 512 = 8192 (total embedding dimension)
- **MLP hidden size**: 32768

### Training/Inference Configuration
- **Batch size**: 1024 (fixed across all experiments)
- **Sequence length**: Not explicitly specified in results
- **Task**: Inference tasks (forward pass evaluation)

## Baseline Configuration
- **Tensor Parallelism (TP)**: Degree 8
- **Pipeline Parallelism (PP)**: Degree 2  
- **Total devices**: TP × PP = 8 × 2 = 16 GPUs
- **Method**: Standard tensor + pipeline parallelism combination

## Proposed Method Configuration
- **Partitioning**: m × n = 16 partitions
- **Mapping**: Each partition assigned to one GPU (total 16 GPUs)
- **Partitioning scheme**: Two-level partitioning as described in methodology

## Evaluation Metrics

### Primary Metrics
1. **Throughput (TPS)**: Tokens processed per second
2. **Time Per Output Token (TPOT)**: Average synchronization and communication overhead per token (milliseconds)

## Results Table

| Model Type | Method | TPS (tokens/sec) | TPOT (ms) |
|------------|--------|------------------|-----------|
| 4-layer Dense | Baseline (TP=8, PP=2) | 1,200,000 | 0.35 |
| 4-layer Dense | Proposed (m×n=16) | 1,580,000 | 0.22 |
| 4-layer MoE | Baseline (TP=8, PP=2) | 850,000 | 0.45 |
| 4-layer MoE | Proposed (m×n=16) | 1,150,000 | 0.30 |

## Performance Analysis

### Dense Model Improvements
- **Throughput increase**: (1,580,000 - 1,200,000) / 1,200,000 = 31.7%
- **Overhead reduction**: (0.35 - 0.22) / 0.35 = 37.1%

### MoE Model Improvements  
- **Throughput increase**: (1,150,000 - 850,000) / 850,000 = 35.3%
- **Overhead reduction**: (0.45 - 0.30) / 0.45 = 33.3%

### Key Findings
1. **Consistent improvements** across both dense and MoE models
2. **Higher gains** for MoE model (35.3% vs 31.7% throughput)
3. **Reduced synchronization cost** evidenced by lower TPOT values
4. **Better hardware utilization** with 16-way partitioning vs 8×2 baseline
5. **Scalability** demonstrated by effective use of all 16 GPUs

## Experimental Validations
- **Precision impact**: FP16 maintains numerical stability while maximizing throughput
- **Batch size impact**: Large batch size (1024) ensures GPU saturation
- **Hardware efficiency**: Performance gains attributed to parallelization strategy rather than hardware idling
- **Communication efficiency**: Reduced cross-device synchronization bandwidth through localized partitioning