# Phase 2: Methodology Extraction

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: Each GPU hosts at most one expert
- **Mathematical formulation**: For E experts and G GPUs, ensure each expert is assigned to distinct GPU if E ≤ G
- **Replication strategy**: If E > G, replicate experts across GPUs to maximize concurrency while balancing memory usage
- **Resource isolation**: Each expert processes tokens without contention from other experts on same device

### 1.2 Cross-Node Distribution Algorithm
- **Topology-aware placement** considers:
  - Node-to-node bandwidth and latency measurements
  - GPU memory capacity per node (H100: 80GB HBM3)
  - Expected token routing patterns based on gating probabilities
- **Optimization objective**: Minimize maximum number of tokens sent across any single link while maintaining one-expert-per-GPU principle
- **Placement algorithm**: 
  1. Construct cluster topology graph with bandwidth weights
  2. Map experts to minimize inter-node communication volume
  3. Balance memory usage across nodes

## 2. Routing and Load Balancing Mechanism

### 2.1 Gating Network Architecture
- **Top-K selection**: K=2 experts activated per token
- **Gating scores**: Softmax over expert logits
- **Load balancing loss**: Auxiliary loss to encourage uniform expert usage

### 2.2 Token Sharding Implementation
- **Token batching**: Group tokens by destination expert to reduce network messages
- **Batch size calculation**: 
  - Input: 1024 sequences × 10000 tokens = 10,240,000 tokens per batch
  - Average per expert: 10,240,000 / 16 = 640,000 tokens
- **Asynchronous routing pipeline**:
  1. Compute gating scores on source GPU
  2. Package tokens by destination expert
  3. Initiate non-blocking send operations
  4. Overlap with local expert computation

### 2.3 Dynamic Load Balancing
- **Monitoring**: Track per-expert token counts every N iterations
- **Adjustment**: Modify gating probabilities using exponential moving average
- **Threshold**: Rebalance when max/min expert load ratio > 1.5

## 3. Communication Overlap and Scheduling

### 3.1 Compute-Communication Overlap Strategy
- **CUDA streams separation**:
  - Stream 0: Expert computation
  - Stream 1: Token communication
- **Double buffering**: 
  - Buffer A: Current tokens being processed
  - Buffer B: Next tokens being received
- **Synchronization points**: 
  - After expert computation completion
  - Before token dispatch to next layer

### 3.2 Pipeline Scheduling Algorithm
- **Layer-wise pipeline**: Each MoE layer treated as micro-stage
- **Token flow**: 
  - Layer 0: Tokens arrive → route to experts → compute → send to Layer 1
  - Overlap: While Layer 0 computes, Layer 1 receives tokens
- **Batch splitting**: Divide 1024 sequences into 4 micro-batches of 256 sequences
- **Pipeline depth**: 4 layers × 4 micro-batches = 16 stages

## 4. Memory Management

### 4.1 Expert Memory Layout
- **Per-expert parameters**:
  - MLP weight 1: (8192, 32768) = 268MB
  - MLP weight 2: (32768, 8192) = 268MB
  - Total per expert: ~536MB
- **Activation memory**:
  - Input tokens: 640,000 × 8192 × 2 bytes = 9.83GB
  - Intermediate activations: 640,000 × 32768 × 2 bytes = 39.3GB
  - Total per expert: ~50GB

### 4.2 Memory Optimization
- **Gradient checkpointing**: Recompute activations during backward pass
- **Mixed precision**: FP16 for compute, FP32 for master weights
- **Memory pooling**: Reuse token buffers across layers

## 5. Integration with Other Parallelism Strategies

### 5.1 Tensor Model Parallelism (TP) Integration
- **Condition**: Applied only when single expert cannot fit on one GPU
- **TP degree**: Optional TP=2 for experts with large hidden sizes
- **Partitioning**: Column-parallel for first linear, row-parallel for second linear

### 5.2 Data Parallelism (DP) Integration
- **DP degree**: 1 (inference-only setting)
- **Weight synchronization**: Not required for inference
- **Future extension**: DP for training with gradient synchronization

### 5.3 Pipeline Parallelism (PP) Integration
- **PP degree**: 4 (one stage per MoE layer)
- **Stage definition**: Each layer = one pipeline stage
- **Communication**: Token routing between stages via NCCL send/recv

## 6. Scalability Parameters

### 6.1 Large EP Regime Definition
- **EP threshold**: EP ≥ 16 qualifies as "large EP"
- **Network requirements**: 
  - Minimum bandwidth: 400 Gbps per GPU (H100 NVLink 4.0)
  - Latency target: <10μs for intra-node, <50μs for inter-node

### 6.2 Scaling Limits
- **Maximum experts**: Limited by cluster GPU count
- **Communication overhead**: O(batch_size × token_dim × experts)
- **Compute efficiency**: >90% GPU utilization achieved at EP=64