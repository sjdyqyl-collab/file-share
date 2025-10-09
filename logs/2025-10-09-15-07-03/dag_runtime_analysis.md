# DAG Runtime Analysis

## Overview
This document analyzes the runtime of two DAG configurations: baseline and FA pool models, focusing on matrix multiplication operations and their parallel/serial execution patterns.

## Baseline Model Analysis

### DAG Structure
- **Total Layers**: 4 transformer layers
- **Parallel Strategy**: Static tensor parallelism (8-way) + pipeline parallelism (2-way)
- **Total GPUs**: 16

### Matrix Multiplication Operations per Layer

#### Attention Block (7 matrix multiplications):
1. **Q Projection**: [4096, 4096] × [batch_size, seq_len, 4096] → [batch_size, seq_len, 4096]
   - Get_Time(4096, 4096, 4096)

2. **K Projection**: [4096, 4096] × [batch_size, seq_len, 4096] → [batch_size, seq_len, 4096]
   - Get_Time(4096, 4096, 4096)

3. **V Projection**: [4096, 4096] × [batch_size, seq_len, 4096] → [batch_size, seq_len, 4096]
   - Get_Time(4096, 4096, 4096)

4. **Attention Score**: [batch_size, 32, seq_len, 128] × [batch_size, 32, 128, seq_len] → [batch_size, 32, seq_len, seq_len]
   - Get_Time(seq_len, 128, seq_len)

5. **Attention Value**: [batch_size, 32, seq_len, seq_len] × [batch_size, 32, seq_len, 128] → [batch_size, 32, seq_len, 128]
   - Get_Time(seq_len, seq_len, 128)

6. **O Projection**: [4096, 4096] × [batch_size, seq_len, 4096] → [batch_size, seq_len, 4096]
   - Get_Time(4096, 4096, 4096)

#### Feed Forward Block (2 matrix multiplications):
7. **MLP1**: [4096, 16384] × [batch_size, seq_len, 4096] → [batch_size, seq_len, 16384]
   - Get_Time(4096, 16384, 4096)

8. **MLP2**: [16384, 4096] × [batch_size, seq_len, 16384] → [batch_size, seq_len, 4096]
   - Get_Time(16384, 4096, 4096)

### Longest Path (Sequential)
```
input → embed → ln1 → {q_proj, k_proj, v_proj} → flash_attn → o_proj → res1 → ln2 → mlp1 → gelu → mlp2 → res2 → comm1 → pipeline_comm → ln3 → {q_proj2, k_proj2, v_proj2} → flash_attn2 → o_proj2 → res3 → ln4 → mlp3 → gelu2 → mlp4 → res4 → layer3 → output
```

**Total Sequential Matrix Multiplications**: 28 (7 per layer × 4 layers)

## FA Pool Model Analysis

### DAG Structure
- **Total Layers**: 4 transformer layers
- **Parallel Strategy**: Dynamic allocation with base layer (8 GPUs) + attention pool (0-32 GPUs)
- **Total Maximum GPUs**: 40
- **Expert Count**: 32 parallel attention experts

### Matrix Multiplication Operations

#### Base Layer Operations (similar to baseline but distributed):
- **QKV Projections**: [4096, 4096] × [batch_size, seq_len, 4096] → [batch_size, seq_len, 4096]
  - Distributed across base GPUs
  - Get_Time(4096, 4096, 4096)

#### Attention Pool Operations (parallel across experts):
- **Per Expert QKV**: [4096, 4096] × [batch_size/32, seq_len, 4096] → [batch_size/32, seq_len, 4096]
  - Get_Time(4096, 4096, 4096/32) × 32 (parallel)

- **Per Expert Attention**: Same pattern as baseline but with reduced batch size
  - Get_Time(seq_len, 128, seq_len) × 32 (parallel)
  - Get_Time(seq_len, seq_len, 128) × 32 (parallel)

- **Per Expert O Projection**: [4096, 4096] × [batch_size/32, seq_len, 4096] → [batch_size/32, seq_len, 4096]
  - Get_Time(4096, 4096, 4096/32) × 32 (parallel)

#### Feed Forward Operations:
- **MLP1**: [4096, 16384] × [batch_size, seq_len, 4096] → [batch_size, seq_len, 16384]
  - Get_Time(4096, 16384, 4096)

- **MLP2**: [16384, 4096] × [batch_size, seq_len, 16384] → [batch_size, seq_len, 4096]
  - Get_Time(16384, 4096, 4096)

### Longest Path (Sequential with Parallel Sections)
```
input → monitor → pool_gate → resource_mgr → split_tokens → 
{q_proj_pool_0..31, k_proj_pool_0..31, v_proj_pool_0..31} → 
{flash_attn_pool_0..31} → {o_proj_pool_0..31} → concat_attn → 
res1 → ffn1 → res2 → ln1 → ffn2 → res3 → ln2 → ffn3 → res4 → ln3 → ffn4 → output
```

**Critical Path Analysis**:
- **Serial Operations**: 28 matrix multiplications (same as baseline)
- **Parallel Operations**: 224 matrix multiplications (7 per expert × 32 experts)
- **Communication Overhead**: Hierarchical reduction across attention pool

## Performance Comparison

### Baseline Model
- **Sequential Runtime**: Σ(Get_Time calls for 28 sequential operations)
- **Parallel Efficiency**: Limited by pipeline stages (2-way) and tensor parallelism (8-way)
- **Communication Overhead**: Between pipeline stages and tensor parallel groups

### FA Pool Model
- **Runtime**: 
  - Base layer: 28 sequential Get_Time calls
  - Attention pool: max(Get_Time calls for parallel experts) + communication overhead
- **Parallel Efficiency**: 
  - Attention computations: 32-way parallel
  - Feed forward: 2-way tensor parallel per layer
- **Communication Overhead**: 
  - Expert selection and token routing
  - Hierarchical reduction for attention outputs

## Key Observations

1. **Matrix Multiplication Count**: 
   - Baseline: 28 sequential operations
   - FA Pool: 28 sequential + 224 parallel operations

2. **Longest Path**: Both models have similar sequential depth, but FA Pool adds parallel computation overhead

3. **Parallelism**: 
   - Baseline: Static 8-way tensor + 2-way pipeline
   - FA Pool: Dynamic 32-way expert + 2-way FFN tensor

4. **Scalability**: FA Pool can utilize up to 40 GPUs vs 16 for baseline

5. **Load Balancing**: FA Pool dynamically allocates resources based on sequence length