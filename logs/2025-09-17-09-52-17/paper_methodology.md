# Methodology: Ring Attention with Sequence Parallelism

## Problem Setup and Notation

### Input Dimensions
- Input sequence: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$
- B: batch size
- L: sequence length  
- d_model: model's hidden size
- H: number of attention heads
- d_h = d_model / H: dimension per head

### MHA Computation
For single attention head:
$$\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_h}} \right) V$$

Where:
- $Q = X W_Q$
- $K = X W_K$  
- $V = X W_V$
- $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_h}$

### Distributed Setting
- P distributed devices: $\{D_0, D_1, \dots, D_{P-1}\}$
- Objective: Compute MHA in parallel with minimal communication overhead and reduced memory footprint

## Sequence Parallelism Implementation

### Data Partitioning
Sequence dimension L split across P devices:
$$X = [X^{(0)}, X^{(1)}, \dots, X^{(P-1)}]$$

Where:
- $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$ resides on device $D_p$
- Each device stores and processes only $\frac{L}{P}$ tokens
- Activation memory reduced by factor of P

### Memory Requirements
- Pre-sequence parallelism: $\mathcal{O}(L d_{\text{model}})$ per device
- Post-sequence parallelism: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per device

## Ring Attention Algorithm

### Ring Topology Structure
- Devices connected in logical ring
- Each device has: previous device (left neighbor) and next device (right neighbor)
- Communication proceeds in P sequential stages

### Algorithm Stages

#### Stage 0: Initialization
Each device computes local projections:
- $Q^{(p)}, K^{(p)}, V^{(p)}$ from local $X^{(p)}$
- Initialize output accumulator: $\text{output}_p = 0$
- Initialize KV_block = $(K^{(p)}, V^{(p)})$

#### Stage 1 to P-1: Ring Communication
For each stage t (0 ≤ t < P):
1. **Compute Partial Attention**:
   - $\text{src} = (p - t) \bmod P$
   - $\text{partial} = \text{Attention}(Q^{(p)}, \text{KV_block})$
   - $\text{output}_p += \text{partial}$

2. **Pass KV Block**:
   - Send KV_block to next device in ring
   - Receive KV_block from previous device

#### Stage P: Aggregation
After P stages:
- Each device has computed attention outputs for its local queries using all keys and values across the entire sequence
- $\text{output}_p$ contains complete attention result for local sequence slice

## Combined Implementation

### Pseudocode
```
for p in parallel on devices:
    # Local projections
    Q_p, K_p, V_p = Project(X_p)
    
    # Initialize
    output_p = 0
    KV_block = (K_p, V_p)
    
    # Ring communication stages
    for t in 0..P-1:
        src_idx = (p - t) mod P
        partial = Attention(Q_p, KV_block)
        output_p += partial
        
        # Async communication
        send KV_block to next device
        receive KV_block from previous device
```

## Communication Complexity Analysis

### Naïve All-Gather Approach
- Each device exchanges: $\mathcal{O}(L d_{\text{model}})$ per step
- Peak bandwidth requirement: High
- Synchronization overhead: Significant

### Ring Attention Approach
- Each device exchanges: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per stage
- Total P stages → same total volume as all-gather
- Peak bandwidth: Lower (distributed over time)
- Better overlap between communication and computation

## Implementation Optimizations

### Communication Primitives
- NCCL send/recv primitives
- MPI point-to-point operations
- Asynchronous communication for overlap

### Precision Optimization
- Mixed-precision: fp16 or bf16 for Q, K, V tensors
- Reduces communication bandwidth requirements
- Maintains numerical stability

### Kernel Fusion
- Fused projection kernels (Q, K, V computation)
- Fused softmax with communication hooks
- Reduces kernel launch overhead

### Memory Management
- Activation checkpointing for memory efficiency
- Stream-based memory allocation
- Overlapping computation with memory transfers