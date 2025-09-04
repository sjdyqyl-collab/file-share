# Phase 2: Methodology Extraction

## Notation and Problem Setup

### Input Dimensions
- Input sequence: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$
  - B: batch size
  - L: sequence length
  - d_model: model's hidden size

### MHA Configuration
- H attention heads
- Each head dimension: $d_h = d_{\text{model}} / H$
- Attention computation for single head: $\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_h}} \right) V$
- Weight matrices: $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_h}$

### Distributed Setup
- P distributed devices: $\{D_0, D_1, \dots, D_{P-1}\}$
- Objective: Compute MHA in parallel with minimal communication overhead and reduced memory footprint

## Sequence Parallelism

### Data Partitioning
- Sequence dimension L split across P devices
- Each device stores: $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$ on device $D_p$
- Memory reduction: Activation memory drops by factor of P from $\mathcal{O}(L d_{\text{model}})$ to $\mathcal{O}(\frac{L}{P} d_{\text{model}})$

### Communication Challenge
- Self-attention requires all keys K and values V across entire sequence
- Naive approach requires all-gather operation (costly for large L)

## Ring Attention

### Ring Topology Structure
- Devices arranged in logical ring
- Communication proceeds in P stages (0 ≤ t < P)

### Algorithm Steps

#### 1. Initialization
- Each device computes local projections:
  - $Q^{(p)}, K^{(p)}, V^{(p)}$ from local $X^{(p)}$

#### 2. Ring Communication (P stages)
- At stage t:
  - Source index: $\text{src} \leftarrow (p - t) \bmod P$
  - Compute partial attention between local $Q^{(p)}$ and current $K^{(\text{src})}, V^{(\text{src})}$
  - Pass $K, V$ tensors to next device in ring
  - Accumulate partial attention results

#### 3. Aggregation
- After P stages, each device has computed attention outputs for local queries using all keys/values

## Combined Ring Attention + Sequence Parallelism

### Integration Strategy
- Sequence parallelism: Defines data placement (sequence split across devices)
- Ring attention: Defines communication order (ring-based instead of all-gather)

### Pseudocode Implementation
```
for p in parallel on devices:
    Q_p, K_p, V_p = Project(X_p)  # Local projections
    output_p = 0
    KV_block = (K_p, V_p)  # Initial KV block
    
    for t in 0..P-1:
        src_idx = (p - t) mod P
        partial = Attention(Q_p, KV_block)  # Compute attention
        output_p += partial  # Accumulate results
        
        # Ring communication
        send KV_block to next device in ring
        receive KV_block from previous device
```

## Communication Complexity Analysis

### Naive All-Gather
- Each device exchanges: $\mathcal{O}(L d_{\text{model}})$ per step
- High peak bandwidth requirement

### Ring Attention
- Each device exchanges: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per stage
- P stages total, same total volume but lower peak bandwidth
- Better overlap between communication and computation

### Memory Benefits
- Sequence parallelism reduces activation memory by factor of P
- Enables processing of longer sequences within memory constraints

## Implementation Details

### Technical Specifications
- **Topology**: NCCL's send/recv primitives or MPI point-to-point operations
- **Overlap**: Computation of attention for one block overlaps with asynchronous communication of next KV block
- **Precision**: Mixed-precision (fp16 or bf16) for Q, K, V to reduce bandwidth
- **Fused Kernels**: Projection and softmax operations fused with communication hooks
- **Scalability**: Performance benefits increase with sequence length L and device count P

### Performance Characteristics
- Optimal for sequences longer than 16k tokens
- Benefits grow with increasing sequence length and number of devices
- Particularly effective in memory-constrained environments