# MoE DAG Runtime Analysis

## Model Specifications
- **Batch Size**: 1024
- **Sequence Length**: 10000
- **Hidden Size**: 8192
- **MLP Hidden Size**: 32768
- **Number of Heads**: 16
- **Head Dimension**: 512
- **Number of Experts**: 16
- **Number of Layers**: 4
- **Precision**: FP16

## Baseline DAG Analysis (TP=8, PP=2)

### Matrix Multiplication Operations

#### 1. Multi-Head Attention (MHA) Components

**QKV Projection**:
- **Operation**: 3 separate matrix multiplications for Q, K, V
- **Dimensions**: [batch_size * sequence_length, hidden_size] × [hidden_size, hidden_size]
- **Shape**: [1024 * 10000, 8192] × [8192, 8192] = [10,240,000, 8192] × [8192, 8192]
- **TP Sharding**: Due to TP=8, each shard processes [10,240,000, 1024] × [1024, 8192]
- **Count**: 3 per attention layer (Q, K, V separately)

**Attention Computation**:
- **Q×K^T**: [batch_size, num_heads, sequence_length, head_dim] × [batch_size, num_heads, head_dim, sequence_length]
- **Shape**: [1024, 16, 10000, 512] × [1024, 16, 512, 10000] = [1024, 16, 10000, 10000]
- **Softmax**: Applied to [1024, 16, 10000, 10000]
- **Attention×V**: [1024, 16, 10000, 10000] × [1024, 16, 10000, 512] = [1024, 16, 10000, 512]

**Output Projection**:
- **Operation**: [batch_size * sequence_length, hidden_size] × [hidden_size, hidden_size]
- **Shape**: [10,240,000, 8192] × [8192, 8192]
- **TP Sharding**: [10,240,000, 1024] × [1024, 8192]

#### 2. Expert Components

**Expert Gate**:
- **Operation**: [batch_size * sequence_length, hidden_size] × [hidden_size, num_experts]
- **Shape**: [10,240,000, 8192] × [8192, 16]

**Expert Linear Layers**:
- **Linear 1**: [tokens_per_expert, hidden_size] × [hidden_size, mlp_hidden_size]
- **Shape**: [tokens_per_expert, 8192] × [8192, 32768]
- **Linear 2**: [tokens_per_expert, mlp_hidden_size] × [mlp_hidden_size, hidden_size]
- **Shape**: [tokens_per_expert, 32768] × [32768, 8192]

**Distribution**: 4 experts per GPU, so each expert processes ~1/4 of tokens
- **Tokens per expert**: ~2,560,000 tokens (assuming uniform distribution)

#### 3. Critical Path Analysis

**Longest Path in Baseline DAG**:
1. **Input → MHA QKV Projection** (parallel across 8 TP shards)
2. **MHA Attention Computation** (parallel across 8 TP shards)
3. **MHA Output Projection** (parallel across 8 TP shards)
4. **MHA All-reduce** (communication across 8 TP shards)
5. **Expert Gate** (routing computation)
6. **Expert Linear 1** (4 experts per GPU in parallel)
7. **Expert Activation** (element-wise, parallel)
8. **Expert Linear 2** (4 experts per GPU in parallel)
9. **Expert Aggregation** (across 4 experts)
10. **Residual connections** (element-wise)
11. **Pipeline Communication** (between stages)
12. **Repeat for Layer 1** (pipeline stage 1)
13. **Pipeline Communication** (stage 0 → stage 1)
14. **Repeat for Layer 2-3** (pipeline stage 1)
15. **Output**

#### 4. Runtime Calculation (Get_Time representation)

**Per Layer Runtime**:
```
T_layer = max(
    // MHA components (parallel across TP=8)
    Get_Time(1024, 10240, 1024, 8192) * 3,  // QKV projections (3 separate)
    Get_Time(1024, 10240, 8192, 1024),      // Output projection
    
    // Expert components (4 experts per GPU, parallel)
    Get_Time(1024, 2560, 8192, 32768) * 4,  // Expert Linear 1 (4 experts)
    Get_Time(1024, 2560, 32768, 8192) * 4,  // Expert Linear 2 (4 experts)
    
    // Communication overhead
    T_allreduce + T_pipeline_comm
)
```

**Total Runtime**:
```
T_baseline = 4 * T_layer + 3 * T_pipeline_comm
```

## Proposed DAG Analysis (EP=64)

### Matrix Multiplication Operations

#### 1. Global Routing
**Global Gate**:
- **Operation**: [batch_size * sequence_length, hidden_size] × [hidden_size, num_experts]
- **Shape**: [10,240,000, 8192] × [8192, 16]

#### 2. Token Distribution
**Token Split**: Non-computational, routing decision

#### 3. Expert Processing (per expert)
Each expert processes a subset of tokens:
- **Average tokens per expert**: ~160,000 (10,240,000/64)

#### 4. MHA within Expert
**QKV Projection**:
- **Operation**: [tokens_per_expert, hidden_size] × [hidden_size, hidden_size * 3]
- **Shape**: [160,000, 8192] × [8192, 24576] (combined QKV)

**Attention Computation**:
- **Q×K^T**: [tokens_per_expert, num_heads, seq_sub, head_dim] × [tokens_per_expert, num_heads, head_dim, seq_sub]
- **Shape**: [160,000/seq_len, 16, seq_sub, 512] × [160,000/seq_len, 16, 512, seq_sub]

**Output Projection**:
- **Operation**: [tokens_per_expert, hidden_size] × [hidden_size, hidden_size]
- **Shape**: [160,000, 8192] × [8192, 8192]

#### 5. Expert MLP
**Expert Gate (within expert)**:
- **Operation**: [tokens_per_expert, hidden_size] × [hidden_size, num_local_experts]
- **Shape**: [160,000, 8192] × [8192, 1] (since each GPU has 1 expert)

**Expert Linear 1**:
- **Operation**: [tokens_per_expert, hidden_size] × [hidden_size, mlp_hidden_size]
- **Shape**: [160,000, 8192] × [8192, 32768]

**Expert Linear 2**:
- **Operation**: [tokens_per_expert, mlp_hidden_size] × [mlp_hidden_size, hidden_size]
- **Shape**: [160,000, 32768] × [32768, 8192]

#### 6. Critical Path Analysis

**Longest Path in Proposed DAG**:
1. **Global Gate** (routing decision)
2. **Token Split** (communication overhead)
3. **Expert Route** (token distribution)
4. **MHA QKV** (within expert, 1 expert per GPU)
5. **MHA Attention** (within expert)
6. **MHA Output** (within expert)
7. **Expert Gate** (within expert)
8. **Expert Linear 1** (within expert)
9. **Expert Activation** (within expert)
10. **Expert Linear 2** (within expert)
11. **Expert Weight** (scaling)
12. **Token Aggregate** (communication overhead)
13. **Residual connections**
14. **Repeat for all 4 layers**
15. **Output**

#### 7. Runtime Calculation (Get_Time representation)

**Per Expert Runtime**:
```
T_expert = max(
    // MHA components (1 expert per GPU)
    Get_Time(1024, 160, 8192, 24576),    // Combined QKV projection
    Get_Time(1024, 160, 8192, 8192),     // Output projection
    
    // Expert MLP (1 expert per GPU)
    Get_Time(1024, 160, 8192, 32768),    // Expert Linear 1
    Get_Time(1024, 160, 32768, 8192),    // Expert Linear 2
    
    // Communication overhead
    T_token_distribution + T_token_aggregation
)
```

**Total Runtime**:
```
T_proposed = 4 * T_expert + 3 * (T_token_distribution + T_token_aggregation)
```

## Comparative Analysis

### Key Differences

1. **Parallelization Strategy**:
   - **Baseline**: TP=8 (tensor parallelism) + PP=2 (pipeline parallelism)
   - **Proposed**: EP=64 (expert parallelism)

2. **Expert Distribution**:
   - **Baseline**: 4 experts per GPU, 16 GPUs total
   - **Proposed**: 1 expert per GPU, 64 GPUs total

3. **Communication Patterns**:
   - **Baseline**: All-reduce within TP groups + pipeline communication
   - **Proposed**: Token routing + aggregation across experts

4. **Matrix Multiplication Granularity**:
   - **Baseline**: Larger matrices (full batch) but sharded across TP
   - **Proposed**: Smaller matrices (per-expert) but more parallel experts

### Runtime Comparison

**Baseline Strengths**:
- Predictable communication patterns
- Lower routing overhead
- Efficient for uniform expert usage

**Proposed Strengths**:
- Massive expert parallelism (64-way)
- Better load balancing with dynamic routing
- Reduced per-GPU memory requirements
- Overlapped computation and communication

**Critical Path Length**:
- **Baseline**: ~4 layers × (MHA + Expert processing + communication)
- **Proposed**: ~4 layers × (routing + expert processing + aggregation)

## Conclusion

The runtime of both DAGs can be expressed as:

**Baseline Runtime**: `4 × max(MHA_time, Expert_time) + communication_overhead`
**Proposed Runtime**: `4 × (routing_time + max_expert_time + aggregation_time)`

Where:
- MHA_time includes TP-sharded QKV projections and attention computation
- Expert_time includes expert-specific MLP computations
- Communication overhead differs significantly between approaches

The actual performance depends on the specific implementation of Get_Time for the given hardware configuration and the efficiency of the communication primitives in each approach.