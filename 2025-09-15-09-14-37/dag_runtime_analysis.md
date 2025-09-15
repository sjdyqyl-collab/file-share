# DAG Runtime Analysis: Baseline vs Proposed MoE Deployment

## Overview
This analysis compares the runtime characteristics of two deployment strategies for large language models:
- **Baseline**: MoE with TP=8, PP=2 using 16 GPUs, 4 experts per GPU
- **Proposed**: Cross-node expert parallelism using 64 GPUs, 1 expert per GPU

## Matrix Multiplication Operations

### Common Parameters
- Batch size: 1024
- Sequence length: 16384
- Hidden size: 8192
- MLP intermediate size: 32768
- Attention heads: 16
- Head dimension: 512

### Baseline DAG Operations

#### Stage 1 (Devices 0-7)
**Multi-Head Attention (MHA):**
1. Query projection: Get_Time(2048×1024, 8192, 8192)
2. Key projection: Get_Time(2048×1024, 8192, 8192)
3. Value projection: Get_Time(2048×1024, 8192, 8192)
4. Attention computation: Get_Time(2048×1024, 16×512, 2048×1024)
5. Output projection: Get_Time(2048×1024, 8192, 8192)

**Gate Network:**
6. Gate projection: Get_Time(2048×1024, 8192, 32)

**Expert Networks** (32 experts total, 4 per GPU):
7. Expert up-projection: Get_Time(64×1024, 8192, 32768)
8. Expert down-projection: Get_Time(64×1024, 32768, 8192)

**Communication:**
9. All-reduce communication across 8 devices

#### Stage 2 (Devices 8-15)
- Identical operations to Stage 1

### Proposed DAG Operations

#### Per Layer (4 layers total)
**Multi-Head Attention with Ring Attention:**
1. Query projection: Get_Time(1024×1024, 8192, 8192)
2. Key projection: Get_Time(1024×1024, 8192, 8192)
3. Value projection: Get_Time(1024×1024, 8192, 8192)
4. Ring attention stages (16 stages): 16 × Get_Time(1024×1024, 16×512, 1024×1024)
5. Output projection: Get_Time(1024×1024, 8192, 8192)

**Gate Network:**
6. Gate projection: Get_Time(1024×1024, 8192, 64)

**Expert Networks** (64 experts total, 1 per GPU):
7. Expert up-projection: Get_Time(16×1024, 8192, 32768)
8. Expert down-projection: Get_Time(16×1024, 32768, 8192)

**Token Operations:**
9. Token routing and aggregation across 64 devices

## Longest Path Analysis

### Baseline DAG Longest Path
```
input → stage1_mha → stage1_gate → stage1_expert_0_0 → stage1_residual → comm_stage1_to_stage2 → stage2_mha → stage2_gate → stage2_expert_8_0 → stage2_residual → output
```

**Critical Path Runtime:**
```
2 × [MHA_time + Expert_time] + Communication_time
= 2 × [5×Get_Time(2048×1024, 8192, 8192) + 2×Get_Time(64×1024, 8192, 32768)] + Communication_time
```

### Proposed DAG Longest Path
```
input → layer1_mha → layer1_gate → layer1_expert_0 → layer1_token_aggregation → layer1_residual → layer2_mha → layer2_gate → layer2_expert_16 → layer2_token_aggregation → layer2_residual → layer3_mha → layer3_gate → layer3_expert_32 → layer3_token_aggregation → layer3_residual → layer4_mha → layer4_gate → layer4_expert_48 → layer4_token_aggregation → layer4_residual → output
```

**Critical Path Runtime:**
```
4 × [Ring_MHA_time + Expert_time + Aggregation_time]
= 4 × [3×Get_Time(1024×1024, 8192, 8192) + 16×Get_Time(1024×1024, 16×512, 1024×1024) + 2×Get_Time(16×1024, 8192, 32768) + Aggregation_time]
```

## Parallelism Comparison

### Baseline
- **Tensor Parallelism**: 8-way within each stage
- **Pipeline Parallelism**: 2 stages
- **Expert Parallelism**: 4 experts per GPU (32 total)
- **Total GPUs**: 16

### Proposed
- **Sequence Parallelism**: 16-way across sequence dimension
- **Expert Parallelism**: 1 expert per GPU (64 total)
- **Ring Attention**: 16 stages for KV exchange
- **Total GPUs**: 64

## Key Insights

1. **Memory Efficiency**: Proposed approach reduces per-device memory by factor of 16 through sequence parallelism
2. **Scalability**: Proposed scales to 64 GPUs vs 16 in baseline
3. **Parallel Expert Processing**: All 64 experts run in parallel in proposed vs 32 in baseline
4. **Communication Patterns**: 
   - Baseline: Pipeline communication between stages
   - Proposed: Ring communication for attention + token routing
5. **Expert Granularity**: Finer-grained expert distribution in proposed (1 expert/GPU vs 4/GPU)

## Runtime Summary

**Baseline Total Runtime:**
```
2 × [Get_Time(2048×1024, 8192, 8192) + Get_Time(2048×1024, 8192, 8192) + Get_Time(2048×1024, 8192, 8192) + Get_Time(2048×1024, 2048×1024, 16×512) + Get_Time(2048×1024, 8192, 8192) + Get_Time(2048×1024, 8192, 32) + Get_Time(64×1024, 8192, 32768) + Get_Time(64×1024, 32768, 8192)] + Communication_time
```

**Proposed Total Runtime:**
```
4 × [Get_Time(1024×1024, 8192, 8192) + Get_Time(1024×1024, 8192, 8192) + Get_Time(1024×1024, 8192, 8192) + 16×Get_Time(1024×1024, 16×512, 1024×1024) + Get_Time(1024×1024, 8192, 8192) + Get_Time(1024×1024, 8192, 64) + Get_Time(16×1024, 8192, 32768) + Get_Time(16×1024, 32768, 8192) + Aggregation_time]
```

Note: The actual numerical runtime values would be obtained by calling Get_Time(m, k, n) with the specified dimensions.