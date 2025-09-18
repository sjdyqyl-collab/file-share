# Methodology - Ring Attention with Sequence Parallelism

## 1. Notation and Problem Setup

### Input Representation
- Input sequence: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$
  - $B$: batch size
  - $L$: sequence length  
  - $d_{\text{model}}$: model's hidden size

### Multi-Head Attention Structure
- $H$ attention heads, each with dimension $d_h = d_{\text{model}} / H$
- Single head attention computation:
  $$\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_h}} \right) V$$
- Query, Key, Value projections:
  $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
  where $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_h}$

### Distributed Setup
- $P$ distributed devices $\{D_0, D_1, \dots, D_{P-1}\}$
- Objective: Compute MHA in parallel with minimal communication overhead and reduced memory footprint

## 2. Sequence Parallelism Method

### Data Partitioning Strategy
- Sequence dimension $L$ split across devices:
  $$X = [X^{(0)}, X^{(1)}, \dots, X^{(P-1)}]$$
- Each device stores: $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$ on device $D_p$
- Memory reduction: Activation memory decreases from $\mathcal{O}(L \cdot d_{\text{model}})$ to $\mathcal{O}(\frac{L}{P} \cdot d_{\text{model}})$

### Communication Challenge
- Self-attention requires all keys $K$ and values $V$ across entire sequence
- Naïve approach requires all-gather operation: costly for large $L$

## 3. Ring Attention Algorithm

### Ring Topology Structure
- Devices arranged in logical ring: $D_0 \rightarrow D_1 \rightarrow \dots \rightarrow D_{P-1} \rightarrow D_0$
- Communication proceeds in $P$ sequential stages

### Algorithm Stages

#### Stage 1: Initialization
Each device computes local projections:
- $Q^{(p)}, K^{(p)}, V^{(p)}$ from local $X^{(p)}$
- $Q^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_h}$
- $K^{(p)}, V^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_h}$

#### Stage 2: Ring Communication (for $t = 0$ to $P-1$)
At each stage $t$:
1. **Source Index Calculation**: $\text{src} \leftarrow (p - t) \bmod P$
2. **Local Computation**: Compute partial attention between $Q^{(p)}$ and current $K^{(\text{src})}, V^{(\text{src})}$:
   $$\text{partial}^{(p,t)} = \text{softmax}\left( \frac{Q^{(p)} (K^{(\text{src})})^\top}{\sqrt{d_h}} \right) V^{(\text{src})}$$
3. **Communication**: Pass $(K^{(\text{src})}, V^{(\text{src})})$ to next device in ring
4. **Accumulation**: Update local output: $\text{output}^{(p)} \leftarrow \text{output}^{(p)} + \text{partial}^{(p,t)}$

#### Stage 3: Final Output
After $P$ stages, each device has:
- $\text{output}^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_h}$: attention output for local sequence slice
- Complete attention context achieved through ring communication

## 4. Combined Algorithm Implementation

### Pseudocode Implementation
```
for p in parallel on devices:
    # Local projections
    Q_p = X_p @ W_Q.T
    K_p = X_p @ W_K.T  
    V_p = X_p @ W_V.T
    
    # Initialize output accumulator
    output_p = zeros(B, L/P, d_h)
    
    # Initialize KV block for ring communication
    KV_block = (K_p, V_p)
    
    # Ring communication stages
    for t in 0..P-1:
        src_idx = (p - t) mod P
        
        # Compute partial attention with current KV block
        partial = softmax(Q_p @ KV_block[0].T / sqrt(d_h)) @ KV_block[1]
        
        # Accumulate results
        output_p += partial
        
        # Ring communication: send to next, receive from previous
        send KV_block to next_device_in_ring
        receive KV_block from previous_device_in_ring
```

## 5. Communication Complexity Analysis

### Communication Volume Comparison
- **Naïve All-Gather**: Each device exchanges $\mathcal{O}(L \cdot d_{\text{model}})$ per step
- **Ring Attention**: Each device exchanges $\mathcal{O}(\frac{L}{P} \cdot d_{\text{model}})$ per stage
- **Total Volume**: Same $\mathcal{O}(L \cdot d_{\text{model}})$ but with **lower peak bandwidth**

### Memory Scaling
- **Sequence Parallelism**: Activation memory per device reduced by factor $P$
- **Communication Buffers**: Only need to store one $\frac{L}{P}$-sized KV block at a time

## 6. Implementation Details

### Technical Specifications
- **Communication Primitives**: NCCL's `send/recv` or MPI point-to-point operations
- **Precision**: Mixed-precision (`fp16` or `bf16`) for Q, K, V tensors
- **Overlap Strategy**: Computation of attention for one block overlaps with asynchronous communication of next KV block
- **Kernel Fusion**: Projection and softmax operations fused with communication hooks
- **Scalability Condition**: Performance benefits increase with $L$ and $P$, particularly for $L > 16\text{k}$ tokens

### Optimization Techniques
- **Asynchronous Communication**: Non-blocking send/receive operations
- **Computation-Communication Overlap**: Pipeline stages to hide communication latency
- **Memory Efficiency**: Minimal buffer allocation for intermediate results
- **Load Balancing**: Equal work distribution across all devices in ring