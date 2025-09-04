# Phase 2: Methodology Extraction

## 1. Problem Setup and Notation

### Input Representation
- Input sequence: $X \in \mathbb{R}^{B \times L \times d_{\text{model}}}$
- Where: $B$ = batch size, $L$ = sequence length, $d_{\text{model}}$ = model hidden size

### Multi-Head Attention Structure
- $H$ attention heads per layer
- Each head dimension: $d_h = d_{\text{model}} / H$
- Attention computation for single head: $\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_h}} \right) V$
- Projections: $Q = X W_Q$, $K = X W_K$, $V = X W_V$ with $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_h}$

### Distributed Setup
- $P$ distributed devices: $\{D_0, D_1, \dots, D_{P-1}\}$
- Objective: Compute MHA in parallel with minimal communication overhead and reduced memory footprint

## 2. Sequence Parallelism Implementation

### Data Partitioning
- Sequence dimension $L$ split across $P$ devices
- Partition: $X = [X^{(0)}, X^{(1)}, \dots, X^{(P-1)}]$
- Each device stores: $X^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$ on device $D_p$
- Memory reduction: Activation memory per device reduced by factor of $P$

### Communication Requirement
- Self-attention requires all keys $K$ and values $V$ across entire sequence
- Creates communication bottleneck if using naive all-gather approach

## 3. Ring Attention Algorithm

### Ring Topology Structure
- Devices arranged in logical ring: $D_0 \rightarrow D_1 \rightarrow \dots \rightarrow D_{P-1} \rightarrow D_0$
- Communication proceeds in $P$ sequential stages

### Algorithm Steps

#### Stage 1: Initialization
- Each device computes local projections:
  - $Q^{(p)}, K^{(p)}, V^{(p)}$ from local $X^{(p)}$
- Initial KV block: $\text{KV_block} = (K^{(p)}, V^{(p)})$ on device $D_p$

#### Stage 2: Ring Communication (for $t = 0$ to $P-1$)
- Source index calculation: $\text{src_idx} = (p - t) \bmod P$
- Each device computes partial attention:
  - $\text{partial} = \text{Attention}(Q^{(p)}, \text{KV_block})$
  - $\text{output}^{(p)} += \text{partial}$
- Communication step:
  - Send KV_block to next device in ring
  - Receive KV_block from previous device

#### Stage 3: Final Aggregation
- After $P$ stages, each device has:
  - Computed attention outputs for its local queries using all keys and values across sequence
  - Final output: $\text{output}^{(p)} \in \mathbb{R}^{B \times \frac{L}{P} \times d_{\text{model}}}$

## 4. Combined Ring Attention + Sequence Parallelism

### Integration Strategy
- **Sequence Parallelism**: Defines data placement - each device stores slice of sequence
- **Ring Attention**: Defines communication order - sequential peer-to-peer exchanges instead of all-gather

### Pseudocode Implementation
```
for p in parallel on devices:
    Q_p, K_p, V_p = Project(X_p)  # Local projections
    output_p = 0  # Initialize accumulator
    KV_block = (K_p, V_p)  # Initial KV from local sequence
    
    for t in 0..P-1:
        src_idx = (p - t) mod P  # Calculate source index
        partial = Attention(Q_p, KV_block)  # Compute attention
        output_p += partial  # Accumulate results
        
        # Ring communication
        send KV_block to next device in ring
        receive KV_block from previous device
```

## 5. Communication Complexity Analysis

### Naive All-Gather Approach
- Each device exchanges: $\mathcal{O}(L d_{\text{model}})$ per step
- Total communication: $\mathcal{O}(P L d_{\text{model}})$
- Peak bandwidth requirement: $\mathcal{O}(L d_{\text{model}})$

### Ring Attention Approach
- Each device exchanges: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ per stage
- Total communication: $\mathcal{O}(L d_{\text{model}})$ across $P$ stages
- Peak bandwidth requirement: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$
- Lower peak bandwidth with better computation-communication overlap

## 6. Implementation Details

### Technical Specifications
- **Communication Primitives**: NCCL send/recv or MPI point-to-point operations
- **Overlap Strategy**: Attention computation for current block overlaps with async communication of next KV block
- **Precision**: Mixed-precision (FP16 or BF16) for Q, K, V tensors to reduce bandwidth
- **Kernel Optimization**: Fused kernels for projection and softmax operations with communication hooks
- **Scalability**: Performance benefits increase with sequence length $L$ and number of devices $P$

### Memory Optimization
- Activation memory per device: $\mathcal{O}(\frac{L}{P} d_{\text{model}})$ instead of $\mathcal{O}(L d_{\text{model}})$
- Memory reduction factor: $P$ (number of devices)
- Particularly effective for $L > 16k$ tokens

### Hardware Requirements
- GPU cluster with NVLink/NVSwitch interconnect
- Minimum 16 GPUs for effective demonstration (as per experiments)
- Support for NCCL or MPI communication libraries