# DAG Runtime Analysis

## Baseline DAG Analysis

### Model Configuration
- Embedding dimension: 8192
- MLP hidden size: 32768
- Batch size: 1024
- Attention heads: 16
- Head dimension: 512
- Tensor parallel degree: 8
- Pipeline parallel degree: 2

### Matrix Multiplication Operations per Layer

#### Attention Block:
1. **QKV Projection** (3 separate matrix multiplications)
   - Dimensions: [1024, 8192] × [8192, 8192]
   - Time: 3 × Get_Time(1024, 8192, 8192)

2. **Attention Computation**:
   - Q×K^T: [1024, 8192] × [8192, 8192] → [1024, 1024]
   - Attention×V: [1024, 1024] × [1024, 8192] → [1024, 8192]
   - Time: Get_Time(1024, 8192, 8192) + Get_Time(1024, 1024, 8192)

3. **Output Projection**:
   - [1024, 8192] × [8192, 8192]
   - Time: Get_Time(1024, 8192, 8192)

#### MLP Block:
1. **First Linear Layer**:
   - [1024, 8192] × [8192, 32768]
   - Time: Get_Time(1024, 8192, 32768)

2. **Second Linear Layer**:
   - [1024, 32768] × [32768, 8192]
   - Time: Get_Time(1024, 32768, 8192)

### Longest Path in Baseline DAG
```
input_0 → layer0_qkv_proj_0 → layer0_attention_0 → layer0_out_proj_0 → layer0_allreduce
          → layer0_residual → layer0_layernorm → layer0_mlp_linear1_0 → layer0_mlp_activation → layer0_mlp_linear2_0 → layer0_mlp_allreduce → layer0_mlp_residual
          → pipeline_send_0_1 → pipeline_recv_1_0 → layer1_qkv_proj_0 → layer1_attention_0 → layer1_out_proj_0 → layer1_allreduce
          → layer1_residual → layer1_layernorm → layer1_mlp_linear1_0 → layer1_mlp_activation → layer1_mlp_linear2_0 → layer1_mlp_allreduce → layer1_mlp_residual → final_output
```

### Total Runtime for Baseline
**2 × [3×Get_Time(1024,8192,8192) + Get_Time(1024,8192,8192) + Get_Time(1024,1024,8192) + Get_Time(1024,8192,8192) + Get_Time(1024,8192,32768) + Get_Time(1024,32768,8192)]**

## Proposed DAG Analysis

### Model Configuration
- Same model size as baseline
- Two-level partitioning: 16 partitions (4 head groups × 4 dimension slices)
- Each partition handles 4 attention heads and 512 dimensions
- Parallel computation across 16 devices

### Matrix Multiplication Operations per Partition

#### Attention Block:
1. **QKV Projection** (3 separate matrix multiplications)
   - Dimensions: [1024, 512] × [512, 512]
   - Time: 3 × Get_Time(1024, 512, 512)

2. **Attention Computation**:
   - Q×K^T: [1024, 512] × [512, 512] → [1024, 512]
   - Attention×V: [1024, 512] × [512, 512] → [1024, 512]
   - Time: Get_Time(1024, 512, 512) + Get_Time(1024, 512, 512)

#### MLP Block:
1. **First Linear Layer**:
   - [1024, 512] × [512, 2048]
   - Time: Get_Time(1024, 512, 2048)

2. **Second Linear Layer**:
   - [1024, 2048] × [2048, 512]
   - Time: Get_Time(1024, 2048, 512)

### Longest Path in Proposed DAG
```
input → broadcast (parallel to all partitions)
      → [q_proj_l1_g0 → attn_l1_g0 → residual_l1_local_0 → mlp_fc_l1_g0 → mlp_gelu_l1_g0 → mlp_proj_l1_g0 → concat_dims_l1_g0] (parallel across 16 partitions)
      → concat_heads_l1 → [similar operations for layer 2] → concat_heads_l2 → output
```

### Total Runtime for Proposed
**Get_Time(1024,512,512) × 5 + Get_Time(1024,512,2048) + Get_Time(1024,2048,512) + communication_overhead**

## Performance Comparison Summary

| Metric | Baseline | Proposed |
|--------|----------|----------|
| Parallelism | Tensor + Pipeline (8×2) | Two-level (16 partitions) |
| Matrix dimensions | Large (8192×8192) | Small (512×512, 512×2048) |
| Sequential steps | 2 full layers | 2 partitioned layers |
| Computation time | O(n³) with large n | O(n³) with much smaller n |
| Communication | Pipeline + Allreduce | Broadcast + Reduce-scatter + All-gather |

The proposed approach achieves significant speedup through massive parallelization, reducing matrix multiplication dimensions from 8192×8192 to 512×512 per partition.