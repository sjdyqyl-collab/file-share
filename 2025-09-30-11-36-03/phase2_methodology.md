# Phase 2: Methodology of FA Pool

## 3. FA Pool Methodology

### 3.1 System Architecture

**Base Layer (8 GPUs)**:
- Contains core model components:
  - Embedding layer (vocab_size=32000, hidden_dim=4096)
  - Positional encoding
  - Output projection layer (4096 → 32000)
  - Feed-forward network (FFN) computations for all 4 layers
- Maintains model state and handles FFN operations
- Dimensions: batch_size × sequence_length × 4096

**Attention Pool (Up to 32 GPUs)**:
- Dedicated to attention computation only
- Activated when sequence_length > 4096 tokens
- Block-wise parallelization strategy
- KV cache sharing across pool GPUs

**Resource Manager**:
- Monitors input sequence length in real-time
- Triggers GPU allocation when threshold exceeded
- Manages synchronization and result aggregation

### 3.2 Dynamic Resource Allocation Strategy

**Threshold Calculation**:
```
Threshold = argmin_t (Attention_Time(t) > FFN_Time + Overhead)
Where:
- Attention_Time(t) = O(t²) for sequence length t
- FFN_Time = constant time for feed-forward computation
- Overhead = communication + synchronization costs
```

**Empirical Threshold**: 4096 tokens (determined through profiling)

**Allocation Process**:
1. **Sequence Length Monitoring**: Real-time monitoring during inference
2. **Threshold Detection**: Compare against 4096 token threshold
3. **Resource Activation**: Allocate p GPUs where p = ceil(sequence_length / 1024)
4. **Maximum Pool Size**: 32 GPUs (empirical limit for diminishing returns)

### 3.3 Attention Parallelization Algorithm

**Block-wise Parallelization**:
```
Input: Query Q (batch, n, 4096), Key K (batch, n, 4096), Value V (batch, n, 4096)
       sequence length n, number of pool GPUs p
Output: Attention output O (batch, n, 4096)

Algorithm:
1. Block size calculation: b = ceil(n / p)
2. For each GPU i in pool (i = 0 to p-1):
   - Extract query block: Q_i = Q[:, i*b:(i+1)*b, :]  # (batch, b, 4096)
   - Full K, V replication: K_i = K, V_i = V         # (batch, n, 4096)
   - Compute local attention: O_i = FlashAttention(Q_i, K, V)
     # Output: O_i (batch, b, 4096)
3. Synchronize across all pool GPUs
4. Concatenate results: O = concat(O_0, O_1, ..., O_p-1)  # (batch, n, 4096)
5. Return final output O
```

**FlashAttention Implementation**:
- Memory-efficient attention computation
- Block size: 128×128 for optimal memory usage
- Avoids materializing full attention matrix (n×n)

### 3.4 Model Layer Distribution

**Layer 0-3 Structure** (each layer):
- **Attention Module**:
  - Query projection: 4096 → 4096 (32 heads × 128 dim/head)
  - Key projection: 4096 → 4096 (32 heads × 128 dim/head)
  - Value projection: 4096 → 4096 (32 heads × 128 dim/head)
  - Output projection: 4096 → 4096
  - **Computation**: Moved to attention pool when active

- **Feed-Forward Network**:
  - First linear: 4096 → 16384 (column-parallel on base layer)
  - Activation: GELU
  - Second linear: 16384 → 4096 (row-parallel on base layer)
  - **Computation**: Always stays on base layer GPUs

### 3.5 Communication Optimization

**KV Cache Sharing**:
- Keys and values (K, V) are fully replicated across all pool GPUs
- Eliminates communication during attention computation
- Memory cost: 2 × batch × n × 4096 per GPU

**Asynchronous Execution**:
- Attention computation overlaps with FFN operations
- Pipeline: Layer i attention (pool) || Layer i-1 FFN (base)
- Reduces total latency by ~15-20%

**Hierarchical Reduction**:
- Tree-based reduction pattern for result aggregation
- Binary tree structure with log₂(p) communication steps
- Bandwidth utilization: NVLink 3.0 (600 GB/s bidirectional)

### 3.6 Memory Distribution

**Base Layer GPUs (8 total)**:
- Model parameters: ~13B parameters × 2 bytes (FP16) = 26GB
- Activations: 4 layers × batch × n × 4096 × 2 bytes = variable
- KV cache: 4 layers × batch × n × 4096 × 2 bytes = variable
- **Total**: 65GB per GPU (at n=8192, batch=1)

**Attention Pool GPUs (up to 32)**:
- Query blocks: batch × b × 4096 × 2 bytes
- Full K,V replication: 2 × batch × n × 4096 × 2 bytes
- Attention computation: 128 × 128 × 2 bytes (FlashAttention blocks)
- **Total**: 45GB per GPU (at n=8192, batch=1)

### 3.7 Synchronization Protocol

**Barrier Synchronization**:
1. Base layer broadcasts Q, K, V to all pool GPUs
2. Pool GPUs compute attention in parallel
3. All pool GPUs complete and signal ready
4. Results concatenated and returned to base layer
5. Base layer continues with FFN computation

**Timing Constraints**:
- Maximum allowed drift: 5ms between fastest and slowest pool GPU
- Automatic load balancing if drift exceeds threshold
- Graceful degradation: reduce pool size if GPUs fail

### 3.8 Model Parameter Dimensions

**Per Layer Parameters**:
- Attention weights:
  - Q_proj: 4096 × 4096 = 16.8M parameters
  - K_proj: 4096 × 4096 = 16.8M parameters
  - V_proj: 4096 × 4096 = 16.8M parameters
  - O_proj: 4096 × 4096 = 16.8M parameters
  - **Total attention**: 67.2M parameters per layer

- FFN weights:
  - W1: 16384 × 4096 = 67.1M parameters
  - W2: 4096 × 16384 = 67.1M parameters
  - **Total FFN**: 134.2M parameters per layer

**Total Model**: 4 × (67.2M + 134.2M) = 805.6M parameters
(Note: Paper mentions ~13B parameters, suggesting additional embedding and output layers)