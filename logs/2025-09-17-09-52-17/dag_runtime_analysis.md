# DAG Runtime Analysis: Baseline vs RA+SP Models

## Executive Summary

This analysis calculates the runtime of two deployment configurations for large language models by examining matrix multiplication operations along the longest paths in their respective DAGs.

## Model Parameters

### Common Architecture Parameters
- **Hidden Size (d_model)**: 8192
- **Attention Heads**: 16
- **Head Dimension**: 512
- **MLP Hidden Size**: 32768
- **Sequence Length**: 10000
- **Batch Size**: 1024
- **Precision**: fp16

## Baseline Model Analysis

### Matrix Multiplication Operations per Layer

#### Attention Block
1. **QKV Projection**: 3 separate matrix multiplications
   - Input: [batch_size × seq_len, hidden_size] = [10,240,000, 8192]
   - Weight: [hidden_size, 3 × hidden_size] = [8192, 24576]
   - Output: [10,240,000, 24576]
   - **Get_Time(10240000, 8192, 24576)**

2. **Attention Output Projection**:
   - Input: [batch_size × seq_len, hidden_size] = [10,240,000, 8192]
   - Weight: [hidden_size, hidden_size] = [8192, 8192]
   - Output: [10,240,000, 8192]
   - **Get_Time(10240000, 8192, 8192)**

#### MLP Block
3. **First Linear Layer**:
   - Input: [batch_size × seq_len, hidden_size] = [10,240,000, 8192]
   - Weight: [hidden_size, mlp_hidden_size] = [8192, 32768]
   - Output: [10,240,000, 32768]
   - **Get_Time(10240000, 8192, 32768)**

4. **Second Linear Layer**:
   - Input: [batch_size × seq_len, mlp_hidden_size] = [10,240,000, 32768]
   - Weight: [mlp_hidden_size, hidden_size] = [32768, 8192]
   - Output: [10,240,000, 8192]
   - **Get_Time(10240000, 32768, 8192)**

### Longest Path Analysis - Baseline

The baseline DAG shows a pipeline-parallel execution with 4 layers divided across 2 pipeline stages:

**Longest Path**: 
`input → layer0_input_split → layer0_ln1_0 → layer0_qkv_0 → layer0_qkv_split → layer0_mha_0 → layer0_attn_out_0 → layer0_attn_allreduce → layer0_residual1 → layer0_ln2_0 → layer0_mlp1_0 → layer0_gelu_0 → layer0_mlp2_0 → layer0_mlp_allreduce → layer0_residual2 → ... → layer3_residual2 → output`

**Critical Path Operations**:
1. **Layer 0**: 4 matrix multiplications (QKV, Attention Output, MLP1, MLP2)
2. **Layer 1**: 4 matrix multiplications
3. **Layer 2**: 4 matrix multiplications  
4. **Layer 3**: 4 matrix multiplications

**Total Sequential Operations**: 16 matrix multiplications

**Baseline Runtime**: 
- 4 × [Get_Time(10240000, 8192, 24576) + Get_Time(10240000, 8192, 8192) + Get_Time(10240000, 8192, 32768) + Get_Time(10240000, 32768, 8192)]

## RA+SP Model Analysis

### Matrix Multiplication Operations per Layer (Per Device)

With sequence parallelism (16-way split):
- **Tokens per device**: 10000/16 = 625 tokens
- **Local batch**: 1024 × 625 = 640,000 tokens per device

#### Attention Block
1. **Local QKV Projection**: 3 separate matrix multiplications
   - Input: [local_batch, hidden_size] = [640,000, 8192]
   - Weight: [hidden_size, 3 × hidden_size] = [8192, 24576]
   - Output: [640,000, 24576]
   - **Get_Time(640000, 8192, 24576)**

2. **Ring Attention Stages**: 16 sequential stages
   - Each stage: [640,000, 512] × [512, 640,000] = attention computation
   - **Get_Time(640000, 512, 640000)** per stage

3. **Attention Output Projection**:
   - Input: [local_batch, hidden_size] = [640,000, 8192]
   - Weight: [hidden_size, hidden_size] = [8192, 8192]
   - Output: [640,000, 8192]
   - **Get_Time(640000, 8192, 8192)**

#### MLP Block
4. **First Linear Layer**:
   - Input: [local_batch, hidden_size] = [640,000, 8192]
   - Weight: [hidden_size, mlp_hidden_size] = [8192, 32768]
   - Output: [640,000, 32768]
   - **Get_Time(640000, 8192, 32768)**

5. **Second Linear Layer**:
   - Input: [local_batch, mlp_hidden_size] = [640,000, 32768]
   - Weight: [mlp_hidden_size, hidden_size] = [32768, 8192]
   - Output: [640,000, 8192]
   - **Get_Time(640000, 32768, 8192)**

### Longest Path Analysis - RA+SP

The RA+SP DAG shows a ring-based execution pattern with 16 devices processing sequence chunks in parallel:

**Longest Path**:
`input → sequence_split → layer0_input_0 → layer0_ln1_0 → layer0_qkv_0 → layer0_split_heads_0 → layer0_ring_stage0_0 → layer0_comm_0_0 → ... → layer0_ring_stage15_0 → layer0_ring_final_0 → layer0_concat_heads_0 → layer0_attn_out_0 → layer0_residual1_0 → layer0_ln2_0 → layer0_mlp1_0 → layer0_gelu_0 → layer0_mlp2_0 → layer0_residual2_0 → ... → layer3_residual2_15 → sequence_gather → output`

**Critical Path Operations**:
1. **Per Layer**: 4 matrix multiplications (reduced dimensions due to sequence parallelism)
2. **Ring Attention**: 16 sequential communication-compute stages
3. **Total Layers**: 4 layers
4. **Parallel Execution**: 16 devices working simultaneously on different sequence chunks

**RA+SP Runtime**:
- 4 × [Get_Time(640000, 8192, 24576) + 16 × Get_Time(640000, 512, 640000) + Get_Time(640000, 8192, 8192) + Get_Time(640000, 8192, 32768) + Get_Time(640000, 32768, 8192)]

## Performance Comparison Summary

### Baseline Model
- **Longest Path**: Sequential execution through 4 layers
- **Matrix Multiplications**: 16 large-scale operations
- **Parallelism**: Limited to tensor parallelism within layers
- **Communication**: All-reduce operations between tensor parallel groups

### RA+SP Model  
- **Longest Path**: Sequential execution through 4 layers with 16-stage ring attention
- **Matrix Multiplications**: 64 operations (4 per layer × 16 devices) but with 16x smaller dimensions
- **Parallelism**: Sequence parallelism + ring attention + tensor parallelism
- **Communication**: Ring-based peer-to-peer communication pattern

### Key Insight
The RA+SP model trades off increased communication stages (16 ring stages vs 4 all-reduce operations) for significantly reduced computation per device (640K vs 10.24M tokens). The critical path shows that while RA+SP has more sequential stages, each operation is much smaller, leading to the expected 20.8% throughput improvement mentioned in the deployment configuration.

## Final Runtime Expressions

**Baseline Runtime**: 4 × [Get_Time(10240000, 8192, 24576) + Get_Time(10240000, 8192, 8192) + Get_Time(10240000, 8192, 32768) + Get_Time(10240000, 32768, 8192)]

**RA+SP Runtime**: 4 × [Get_Time(640000, 8192, 24576) + 16 × Get_Time(640000, 512, 640000) + Get_Time(640000, 8192, 8192) + Get_Time(640000, 8192, 32768) + Get_Time(640000, 32768, 8192)]