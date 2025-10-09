# FA Pool Methodology - Detailed Technical Specification

## 3. FA Pool Methodology

### 3.1 System Architecture

FA Pool operates on the principle of dynamic resource allocation based on computational demand. The system architecture consists of:

**Base Layer**: The primary computational layer containing the model's core components:
- Embedding layer (4096 dimensions)
- Positional encoding
- Output projection layer
- 4 transformer layers with FFN computations
- Total parameters: ~13B

**Attention Pool**: A dynamically allocated set of GPUs dedicated to attention computation:
- Minimum: 0 GPUs (when sequence ≤ 4096 tokens)
- Maximum: 32 GPUs
- GPU type: NVIDIA A100 80GB

**FFN Layer**: Feed-forward network computations that remain on the base layer:
- Hidden dimension: 16384 (4× hidden size)
- Activation: GELU
- Parameters per layer: ~67M

**Resource Manager**: Monitors sequence length and allocates/deallocates GPU resources for the attention pool:
- Monitoring frequency: Per batch
- Allocation latency: <10ms
- Deallocation latency: <5ms

### 3.2 Dynamic Resource Allocation Strategy

The FA Pool strategy operates through the following mechanism:

1. **Sequence Length Monitoring**: Continuously monitor input sequence length during inference
   - Input: Token sequence of length n
   - Measurement: Number of tokens after tokenization
   - Update frequency: Per forward pass

2. **Threshold Detection**: Compare sequence length against predefined thresholds
   - Primary threshold: 4096 tokens
   - Secondary thresholds: 8192, 16384, 32768 tokens (for pool size scaling)
   - Hysteresis: 256 tokens to prevent oscillation

3. **Resource Activation**: When sequence length exceeds the threshold, activate additional GPUs for the attention pool
   - Activation mapping:
     - 4096 < n ≤ 8192: 8 pool GPUs
     - 8192 < n ≤ 16384: 16 pool GPUs
     - 16384 < n ≤ 32768: 24 pool GPUs
     - n > 32768: 32 pool GPUs

4. **Workload Distribution**: Partition attention computation across the available pool GPUs
   - Partitioning strategy: Block-wise along sequence dimension
   - Block size: b = ceil(n / p) where p is number of pool GPUs
   - Overlap: 64 tokens for boundary handling

5. **Result Aggregation**: Collect and synchronize results from pool GPUs
   - Reduction method: Tree-based hierarchical reduction
   - Communication pattern: All-gather with 2-step hierarchy
   - Synchronization: CUDA streams with events

6. **Resource Deactivation**: Release pool resources when sequence length drops below threshold
   - Deactivation delay: 5 seconds (configurable)
   - Graceful shutdown: Complete current batch before deactivation

### 3.3 Attention Parallelization

Within the attention pool, we implement a block-wise parallelization strategy:

```
Input: Query Q, Key K, Value V, sequence length n, number of pool GPUs p
Output: Attention output O

Algorithm Parameters:
- Hidden dimension: d = 4096
- Attention heads: h = 32
- Head dimension: d_h = d / h = 128
- Block size: b = ceil(n / p)

1. Block size calculation: b = ceil(n / p)
2. For each GPU i in pool:
   - Extract query block: Q_i = Q[i*b:(i+1)*b]  # Shape: (b, d)
   - Extract key block: K_i = K[i*b:(i+1)*b]    # Shape: (b, d)
   - Extract value block: V_i = V[i*b:(i+1)*b]  # Shape: (b, d)
   - Compute local attention: O_i = FlashAttention(Q_i, K, V)
     - Uses full K, V matrices (replicated across GPUs)
     - Computes attention for query block against entire sequence
     - Output shape: (b, d)
3. Synchronize and aggregate results: O = concat(O_0, O_1, ..., O_p-1)
   - Concatenation along sequence dimension
   - Final output shape: (n, d)
4. Return final output O
```

### 3.4 Communication Optimization

To minimize communication overhead, FA Pool implements:

**KV Cache Sharing**: 
- Keys and values are replicated across pool GPUs
- Replication strategy: Broadcast during preprocessing phase
- Memory overhead: O(n×d×p) where p is number of pool GPUs
- Communication pattern: NCCL broadcast operation

**Asynchronous Execution**:
- Attention computation overlaps with FFN operations on base layer
- CUDA streams: Separate streams for computation and communication
- Event synchronization: Ensure data dependencies are met
- Overlap efficiency: 75-80% of attention time overlapped with FFN

**Hierarchical Reduction**:
- Tree-based reduction pattern to minimize communication steps
- Two-level hierarchy: Within-node (NVLink) and across-node (InfiniBand)
- Reduction steps: log₂(p) where p is number of pool GPUs
- Bandwidth utilization: 85-90% of theoretical peak

### 3.5 Threshold Determination

The sequence length threshold is determined through empirical analysis:

**Threshold Formula**:
```
Threshold = argmin_t (Attention_Time(t) > FFN_Time + Overhead)

Where:
- Attention_Time(t) = α × t² + β × t + γ
- FFN_Time = constant (measured 45ms for 4-layer model)
- Overhead = Communication + Synchronization + Allocation
- α, β, γ: Empirical coefficients from profiling
```

**Empirical Values**:
- α = 2.3 × 10⁻⁶ ms/token²
- β = 1.1 × 10⁻³ ms/token
- γ = 12.5 ms
- Overhead = 15ms (for 8 pool GPUs)
- **Calculated threshold: 4096 tokens**

### 3.6 Model Parameter Distribution

**Base Layer (8 GPUs)**:
- Embedding layer: Replicated across all 8 GPUs
- Layer norms: Replicated across all 8 GPUs
- FFN layers: Tensor parallel across 8 GPUs (column-row split)
- Output layer: Tensor parallel across 8 GPUs

**Attention Pool (up to 32 GPUs)**:
- Query projection: Split across pool GPUs (row parallel)
- Key projection: Replicated across pool GPUs
- Value projection: Replicated across pool GPUs
- Output projection: Split across pool GPUs (column parallel)

### 3.7 Memory Layout

**Base Layer Memory (per GPU)**:
- Model parameters: 8.125GB (13B / 8 / 2 for tensor parallel)
- Activations: 32GB (varies with batch size)
- KV cache: 24GB (for sequence length up to 32K)
- Total: ~65GB

**Attention Pool Memory (per GPU)**:
- Model parameters: 2.031GB (attention-specific parameters)
- Activations: 20GB (block-wise computation)
- KV cache: 22GB (replicated keys/values)
- Total: ~45GB

### 3.8 Implementation Details

**CUDA Kernels**:
- FlashAttention v2 for attention computation
- Custom NCCL-based communication kernels
- Asynchronous memory copy operations

**Framework**:
- PyTorch 2.0 with CUDA 12.0
- NCCL 2.18 for communication
- Custom CUDA extensions for FlashAttention

**Precision**:
- Compute: FP16/BF16 mixed precision
- Communication: FP16
- Master weights: FP32

### 3.9 Fault Tolerance

**Pool GPU Failure Handling**:
- Redundant computation: Failed GPU blocks recomputed on remaining GPUs
- Checkpointing: Periodic attention state checkpointing
- Recovery time: <100ms for single GPU failure

**Load Balancing**:
- Dynamic block size adjustment based on GPU performance
- Work stealing for uneven load distribution
- Performance monitoring every 100ms