# Phase 2: Methodology Extraction - FA Pool Paper

## 3.1 System Architecture

### Base Layer
- **Function**: Primary computational layer containing core model components
- **Components**: Embedding layer, positional encoding, output layers
- **GPU Allocation**: 8 GPUs (fixed)
- **Responsibility**: Maintains model coherence and handles non-attention operations

### Attention Pool
- **Function**: Dynamically allocated GPUs dedicated to attention computation
- **GPU Allocation**: Up to 32 additional GPUs (dynamic)
- **Activation**: Triggered when sequence length exceeds threshold (4096 tokens)
- **Deactivation**: Released when sequence length drops below threshold

### FFN Layer
- **Location**: Remains on base layer
- **Rationale**: Feed-forward networks have linear complexity and don't require parallelization
- **Benefit**: Reduces communication overhead by keeping FFN operations local

### Resource Manager
- **Function**: Monitors sequence length and manages GPU allocation
- **Operations**: 
  - Continuous sequence length monitoring during inference
  - Threshold detection and comparison
  - Resource activation/deactivation
  - Workload distribution coordination

## 3.2 Dynamic Resource Allocation Strategy

### Mechanism Steps
1. **Sequence Length Monitoring**: Continuous monitoring during inference
2. **Threshold Detection**: Compare against predefined threshold (4096 tokens)
3. **Resource Activation**: Activate additional GPUs when threshold exceeded
4. **Workload Distribution**: Partition attention computation across pool GPUs
5. **Result Aggregation**: Collect and synchronize results from pool GPUs
6. **Resource Deactivation**: Release resources when sequence length drops

### Threshold Logic
```
Threshold = argmin_t (Attention_Time(t) > FFN_Time + Overhead)
```
Where Overhead accounts for communication and synchronization costs

## 3.3 Attention Parallelization

### Block-wise Parallelization Strategy
```
Input: Query Q, Key K, Value V, sequence length n, number of pool GPUs p
Output: Attention output O

1. Block size calculation: b = ceil(n / p)
2. For each GPU i in pool:
   - Extract block: Q_i = Q[i*b:(i+1)*b], K_i = K[i*b:(i+1)*b], V_i = V[i*b:(i+1)*b]
   - Compute local attention: O_i = FlashAttention(Q_i, K, V)
3. Synchronize and aggregate results: O = concat(O_0, O_1, ..., O_p-1)
4. Return final output O
```

### Key Features
- **Block Size**: Calculated as ceiling of sequence length divided by pool GPUs
- **Local Computation**: Each GPU computes attention for its assigned block
- **Flash Attention**: Uses memory-efficient FlashAttention algorithm
- **Result Aggregation**: Concatenation of results from all pool GPUs

## 3.4 Communication Optimization

### KV Cache Sharing
- **Strategy**: Keys and values replicated across pool GPUs
- **Benefit**: Avoids communication during attention computation
- **Trade-off**: Increased memory usage for reduced communication

### Asynchronous Execution
- **Strategy**: Attention computation overlaps with FFN operations
- **Benefit**: Hides communication latency
- **Implementation**: Parallel execution pipelines

### Hierarchical Reduction
- **Pattern**: Tree-based reduction for result aggregation
- **Benefit**: Minimizes communication steps
- **Complexity**: O(log p) steps for p GPUs

## 3.5 Implementation Details

### Model Configuration
- **Layers**: 4 transformer layers
- **Hidden Dimension**: 4096
- **Attention Heads**: 32
- **Feed-forward Dimension**: 16384
- **Model Parameters**: ~13B parameters
- **Activation Function**: GELU
- **Normalization**: Pre-norm with RMSNorm

### Hardware Requirements
- **GPU Model**: NVIDIA A100 80GB
- **Interconnect**: NVLink 3.0 and InfiniBand
- **Minimum Configuration**: 8 base GPUs + up to 32 pool GPUs
- **Memory per GPU**: 65GB (base), 45GB (pool)

### Baseline Comparison
- **Baseline Strategy**: TP=8, PP=2 (16 GPUs total)
- **FA Pool Strategy**: 8 base GPUs + up to 32 pool GPUs
- **Resource Advantage**: FA Pool uses more total GPUs but achieves better utilization