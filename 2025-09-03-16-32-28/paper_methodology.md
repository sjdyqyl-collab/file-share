# Methodology - Ring Attention with Sequence Parallelism

## Problem Setup and Notation

### Input Dimensions
- Input sequence: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$
  - $B$: batch size
  - $L$: sequence length
  - $d_{\text{model}}$: model's hidden size

### Multi-Head Attention Structure
- $H$ attention heads per layer
- Each head dimension: $d_h = d_{\text{model}} / H$
- Attention computation for single head: $\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_h}} \right) V$

### Weight Matrices
- $Q = X W_Q$
- $K = X W_K$
- $V = X W_V$
- Weight matrices: $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_h}$

### Distributed Setup
- $P$ distributed devices: $\{D_0, D_1, \dots, D_{P-1}\}$
- Objective: Compute MHA in parallel with minimal communication overhead and reduced memory footprint

## Sequence Parallelism Implementation

### Data Partitioning
- Sequence dimension $L$ split across $P$ devices:
- $X = [X^{(0)}, X^{(1)}, \dots, X^{(P-1)}]$
- Each device $D_p$ stores: $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$

### Memory Reduction
- Activation memory per device: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$
- Total memory reduction factor: $P$
- Challenge: Self-attention requires all keys $K$ and values $V$ across entire sequence

## Ring Attention Algorithm

### Ring Topology Structure
- Devices arranged in logical ring: $D_0 \rightarrow D_1 \rightarrow \dots \rightarrow D_{P-1} \rightarrow D_0$
- Communication pattern: Sequential peer-to-peer exchanges

### Algorithm Stages

#### Stage 1: Initialization
- Each device computes local projections:
  - $Q^{(p)} = X^{(p)} W_Q$
  - $K^{(p)} = X^{(p)} W_K$
  - $V^{(p)} = X^{(p)} W_V$

#### Stage 2: Ring Communication (P stages)
For each stage $t$ where $0 \leq t < P$:
1. **Source Index Calculation**: $\text{src} = (p - t) \bmod P$
2. **Current KV Block**: Device holds $K^{(\text{src})}, V^{(\text{src})}$
3. **Partial Attention Computation**:
   - Compute: $\text{partial}^{(p,t)} = \text{softmax}\left( \frac{Q^{(p)} (K^{(\text{src})})^\top}{\sqrt{d_h}} \right) V^{(\text{src})}$
4. **Accumulation**: $O^{(p)} += \text{partial}^{(p,t)}$
5. **Ring Communication**:
   - Send current $(K^{(\text{src})}, V^{(\text{src})})$ to next device
   - Receive new $(K, V)$ block from previous device

#### Stage 3: Final Output
- After $P$ stages, each device $D_p$ has:
  - $O^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$: attention output for local sequence slice

## Combined Ring Attention + Sequence Parallelism

### Integration Strategy
- **Sequence Parallelism**: Defines data placement (sequence split across devices)
- **Ring Attention**: Defines communication order (ring-based KV exchange)

### Pseudocode Implementation
```
for p in parallel on devices:
    # Local projection
    Q_p, K_p, V_p = Project(X_p)
    
    # Initialize output accumulator
    output_p = 0
    
    # Initial KV block
    KV_block = (K_p, V_p)
    
    # Ring attention stages
    for t in 0..P-1:
        # Determine source index
        src_idx = (p - t) mod P
        
        # Compute partial attention
        partial = Attention(Q_p, KV_block)
        output_p += partial
        
        # Ring communication
        send KV_block to next device in ring
        receive KV_block from previous device
```

## Communication Analysis

### Bandwidth Requirements
- **Naïve All-Gather**: Each device exchanges $\mathcal{O}(L d_{\text{model}})$ per step
- **Ring Attention**: Each device exchanges $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per stage
- **Total Volume**: Same for both methods, but Ring Attention has:
  - Lower peak bandwidth requirements
  - Better overlap between communication and computation

### Memory Footprint
- **Without Sequence Parallelism**: $\mathcal{O}(L d_{\text{model}})$ per device
- **With Sequence Parallelism**: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per device
- **Memory Reduction Factor**: $P$

## Implementation Details

### Communication Primitives
- **Backend**: NCCL's `send/recv` primitives or MPI point-to-point operations
- **Topology**: Logical ring over physical interconnect

### Optimization Techniques
- **Overlap**: Computation of attention for one block overlaps with asynchronous communication of next KV block
- **Precision**: Mixed-precision (FP16 or BF16) for Q, K, V tensors to reduce bandwidth
- **Fused Kernels**: Projection and softmax operations fused with communication hooks
- **Kernel Launch**: Reduced overhead through fusion

### Scalability Characteristics
- Performance benefits increase with:
  - Sequence length $L$ (especially $L > 16K$ tokens)
  - Number of devices $P$
  - Model size (due to memory constraints)