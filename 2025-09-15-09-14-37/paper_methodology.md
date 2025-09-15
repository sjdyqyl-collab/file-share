# Ring Attention with Sequence Parallelism: Detailed Methodology

## Problem Setup and Notation

### Input Dimensions
- Input sequence: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$
- Batch size: $B$
- Sequence length: $L$
- Model hidden size: $d_{\text{model}}$
- Number of attention heads: $H$
- Head dimension: $d_h = d_{\text{model}} / H$

### Multi-Head Attention Computation
For each attention head:
- Query: $Q = X W_Q$ where $W_Q \in \mathbb{R}^{d_{\text{model}} \times d_h}$
- Key: $K = X W_K$ where $W_K \in \mathbb{R}^{d_{\text{model}} \times d_h}$
- Value: $V = X W_V$ where $W_V \in \mathbb{R}^{d_{\text{model}} \times d_h}$

Single head attention:
$$\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_h}} \right) V$$

### Distributed Setup
- $P$ distributed devices: $\{D_0, D_1, \dots, D_{P-1}\}$
- Objective: Compute MHA in parallel with minimal communication overhead and reduced memory footprint

## Sequence Parallelism Methodology

### Data Partitioning
The sequence dimension $L$ is split across $P$ devices:
$$X = [X^{(0)}, X^{(1)}, \dots, X^{(P-1)}]$$

Where:
- $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$ resides on device $D_p$
- Each device stores and processes only $\frac{L}{P}$ tokens
- Memory reduction: Activation memory reduced by factor of $P$

### Memory Requirements
- **Without sequence parallelism**: $\mathcal{O}(L \cdot d_{\text{model}})$ per device
- **With sequence parallelism**: $\mathcal{O}(\frac{L}{P} \cdot d_{\text{model}})$ per device

## Ring Attention Methodology

### Ring Topology Structure
Devices arranged in logical ring: $D_0 \rightarrow D_1 \rightarrow \dots \rightarrow D_{P-1} \rightarrow D_0$

### Algorithm Stages
The algorithm proceeds in $P$ stages:

#### Stage 0: Initialization
Each device computes local projections:
- $Q^{(p)}, K^{(p)}, V^{(p)}$ from local $X^{(p)}$
- Initialize output accumulator: $\text{output}_p = 0$

#### Stages 1 to P: Ring Communication
For each stage $t$ where $0 \leq t < P$:

1. **Source Index Calculation**:
   $$\text{src} = (p - t) \bmod P$$

2. **Local Computation**:
   - Compute partial attention between local $Q^{(p)}$ and current $K^{(\text{src})}, V^{(\text{src})}$
   - Accumulate result: $\text{output}_p += \text{Attention}(Q^{(p)}, K^{(\text{src})}, V^{(\text{src})})$

3. **Communication**:
   - Send current $(K^{(\text{src})}, V^{(\text{src})})$ to next device in ring
   - Receive new $(K, V)$ block from previous device

### Mathematical Formulation
For device $D_p$ at stage $t$:
- Current KV source: $(p - t) \bmod P$
- Attention computation: $\text{softmax}\left(\frac{Q^{(p)} (K^{(\text{src})})^\top}{\sqrt{d_h}}\right) V^{(\text{src})}$

## Combined Ring Attention + Sequence Parallelism

### Integration Strategy
1. **Sequence parallelism** defines data placement across devices
2. **Ring attention** defines communication pattern for KV exchange
3. **Combined approach** eliminates all-gather operations

### Complete Algorithm
```
for p in parallel on devices D_0 to D_{P-1}:
    # Local projection
    Q_p, K_p, V_p = LinearProjection(X_p)
    
    # Initialize output accumulator
    output_p = zeros(B, L/P, d_model)
    
    # Initialize KV block with local values
    KV_block = (K_p, V_p)
    
    # Ring communication loop
    for t in 0 to P-1:
        # Determine source device
        src_idx = (p - t) mod P
        
        # Compute partial attention
        partial_output = MultiHeadAttention(Q_p, KV_block[0], KV_block[1])
        
        # Accumulate results
        output_p += partial_output
        
        # Non-blocking send/receive
        send_async(KV_block, to=(p+1) mod P)
        KV_block = recv_async(from=(p-1) mod P)
        
        # Synchronize before next iteration
        sync()
```

## Communication Complexity Analysis

### Naïve All-Gather Approach
- **Per step**: Each device exchanges $\mathcal{O}(L \cdot d_{\text{model}})$
- **Peak bandwidth**: High due to simultaneous all-to-all communication
- **Synchronization**: Global barrier required

### Ring Attention Approach
- **Per stage**: Each device exchanges $\mathcal{O}(\frac{L}{P} \cdot d_{\text{model}})$
- **Total volume**: $P \times \mathcal{O}(\frac{L}{P} \cdot d_{\text{model}}) = \mathcal{O}(L \cdot d_{\text{model}})$
- **Peak bandwidth**: Lower due to sequential communication pattern
- **Overlap**: Computation overlaps with communication

## Implementation Details

### Communication Primitives
- **Backend**: NCCL `send/recv` primitives or MPI point-to-point operations
- **Topology**: Logical ring over physical interconnect
- **Synchronization**: CUDA streams for async operations

### Performance Optimizations
1. **Mixed Precision**: FP16/BF16 for Q, K, V tensors
2. **Kernel Fusion**: Fused projection and softmax operations
3. **Communication Overlap**: Async send/recv with computation
4. **Load Balancing**: Equal sequence partitioning across devices

### Memory Layout
- **Input**: $X^{(p)}$ stored as [B, L/P, d_model] tensor
- **Projections**: Q, K, V as [B, L/P, H, d_h] tensors
- **KV Cache**: Temporary storage for received blocks
- **Output**: Accumulated attention output [B, L/P, d_model]

## Scalability Considerations

### Scaling Factors
- **Sequence length L**: Benefits increase with longer sequences
- **Device count P**: Linear memory reduction, sub-linear communication overhead
- **Model size**: Benefits independent of parameter count
- **Head count H**: Parallelizable across heads within each device

### Critical Thresholds
- **Minimum benefit**: L > 16k tokens for significant improvements
- **Optimal P**: Balance between memory reduction and communication overhead
- **Hardware**: Requires high-bandwidth interconnect (NVLink/NVSwitch)