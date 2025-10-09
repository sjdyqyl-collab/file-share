# FA Pool Methodology - Technical Details

## 3. FA Pool Methodology (Detailed)

### 3.1 System Architecture Components

**Base Layer Configuration**:
- Contains: embedding, positional encoding, output layers
- GPU Count: 8 GPUs (fixed)
- Memory per GPU: 65GB
- Components: Model backbone, FFN layers

**Attention Pool Configuration**:
- Dynamic GPU allocation: 0-32 GPUs
- Memory per GPU: 45GB (reduced due to block-wise computation)
- Activation threshold: 4096 tokens
- Maximum pool size: 32 GPUs

**Resource Manager**:
- Monitors sequence length in real-time
- Triggers GPU allocation/deallocation
- Manages KV cache sharing across pool GPUs

### 3.2 Dynamic Resource Allocation Algorithm

```
Input: Sequence length n, threshold T=4096, max_pool_size=32
Process:
1. if n <= T:
   - pool_size = 0
   - attention_computed_on_base_layer()
2. else:
   - pool_size = min(ceil(n/2048), max_pool_size)
   - activate_gpus(pool_size)
   - distribute_attention_computation()
3. return pool_size, gpu_allocation
```

### 3.3 Attention Parallelization Details

**Block-wise Parallelization Parameters**:
- Block size calculation: b = ceil(n / p) where p = pool_size
- Each GPU i processes: Q_i = Q[i*b:(i+1)*b], K_i = K[i*b:(i+1)*b], V_i = V[i*b:(i+1)*b]
- Local attention: O_i = FlashAttention(Q_i, K, V)
- Result aggregation: O = concat(O_0, O_1, ..., O_p-1)

**KV Cache Sharing Strategy**:
- Keys (K) and Values (V) are replicated across all pool GPUs
- Size per GPU: n × d_model × num_heads
- Avoids communication during attention computation

### 3.4 Communication Optimization

**Hierarchical Reduction Pattern**:
- Tree-based reduction with log(p) steps
- Each step reduces communication by 50%
- Final aggregation on base layer

**Asynchronous Execution**:
- Attention computation overlaps with FFN operations
- Pipeline depth: 2 stages (attention + FFN)
- Communication latency hidden through overlap

### 3.5 Threshold Determination Formula

**Threshold Calculation**:
```
Threshold = argmin_t (Attention_Time(t) > FFN_Time + Overhead)
where:
- Attention_Time(t) = O(t² × d_model × num_heads / (base_gpus + pool_gpus))
- FFN_Time = O(d_model × d_ffn × sequence_length / base_gpus)
- Overhead = communication_cost + synchronization_cost
```

**Empirical Threshold**: 4096 tokens (determined through profiling)

### 3.6 Model Parameters for Deployment

**Layer Configuration**:
- Number of layers: 4
- Hidden dimension: 4096
- Attention heads: 32
- Head dimension: 128 (4096/32)
- Feed-forward dimension: 16384 (4× hidden dimension)
- Attention dropout: 0.1
- Activation: GELU
- Normalization: Pre-norm RMSNorm

**Memory Requirements**:
- Model parameters: ~13B parameters
- Parameter memory: 52GB (13B × 4 bytes)
- Activation memory: Variable based on sequence length
- KV cache: n × d_model × num_layers

### 3.7 GPU Mapping Strategy

**Base Layer (8 GPUs)**:
- GPUs 0-7: Model backbone + FFN layers
- Tensor parallelism: 8-way within base layer
- Pipeline stage 1: Layers 0-1
- Pipeline stage 2: Layers 2-3

**Attention Pool (0-32 GPUs)**:
- GPUs 8-39: Dynamic allocation
- Each GPU handles: 1/p of attention computation
- No tensor parallelism within pool (embarrassingly parallel)
- Direct memory access to base layer for KV values

### 3.8 Synchronization Protocol

**Step 1**: Base layer computes embeddings and positional encodings
**Step 2**: If sequence > threshold, activate attention pool
**Step 3**: Distribute Q, K, V to pool GPUs
**Step 4**: Parallel attention computation across pool
**Step 5**: Hierarchical reduction of attention outputs
**Step 6**: FFN computation on base layer (overlapped with step 4-5)
**Step 7**: Output layer computation on base layer