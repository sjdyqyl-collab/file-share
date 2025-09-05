# Methodology - Large-Scale Cross-Node Expert Parallelism

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Constraint**: Each GPU hosts at most one expert
- **Assignment rule**: 
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Replicate experts across GPUs while maintaining memory balance
- **Benefit**: Eliminates intra-GPU contention between experts

### 1.2 Cross-Node Distribution Algorithm
- **Inputs**: 
  - Number of experts E
  - Number of GPUs G
  - Cluster topology (bandwidth matrix, latency matrix)
  - GPU memory capacity per node
- **Objective**: Minimize max tokens sent across any single link
- **Constraints**: 
  - One-expert-per-GPU principle
  - Memory capacity per GPU
  - Balanced load across nodes

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- **Top-K routing**: K=2 for each token
- **Gating network**: Standard softmax over expert scores
- **Expert capacity**: Fixed capacity factor to prevent overflow

### 2.2 Token Sharding Process
1. **Token grouping**: 
   - Group tokens by destination expert ID
   - Batch size per expert: 1024 tokens max
2. **Asynchronous routing**: 
   - Send token batches using NCCL/MPI
   - Overlap with current expert computation
3. **Load balancing**: 
   - Monitor per-expert load via token counts
   - Adjust gating probabilities: P_i = softmax(score_i + load_penalty_i)
   - Load penalty: λ × (current_load_i / avg_load)

## 3. Communication Overlap and Scheduling

### 3.1 Compute-Communication Overlap
- **CUDA streams**: 
  - Stream 0: Expert computation
  - Stream 1: Token communication
- **NCCL operations**: 
  - ncclAllToAllv for token exchange
  - Double buffering: Buffer A (compute) / Buffer B (communicate)

### 3.2 Pipeline Scheduling
- **Layer-wise pipeline**: 
  - Each MoE layer = 1 pipeline stage
  - 4 stages for 4-layer model
- **Token flow**: 
  - Stage i sends output tokens to Stage i+1 immediately
  - Partial batch processing: Start computation with 256 tokens, receive remaining 768

## 4. Memory Management

### 4.1 Expert Memory Layout
- **Expert parameters**: 
  - MLP weights: 2 × (hidden × d_model) = 2 × (32768 × 8192) = 536MB
  - Expert size: ~537MB per expert (FP16)
- **Token buffer**: 
  - Input tokens: 1024 × 8192 × 2 bytes = 16MB
  - Output tokens: 1024 × 8192 × 2 bytes = 16MB

### 4.2 Memory Optimization
- **Gradient checkpointing**: Not used (inference only)
- **Activation recomputation**: Not applicable
- **Zero redundancy**: Each GPU stores only its assigned expert

## 5. Parallelism Integration

### 5.1 Expert Parallelism (EP)
- **EP degree**: 16 (minimum for large EP)
- **EP groups**: 1 group per layer
- **Expert assignment**: 1 expert per GPU × 16 GPUs = 16 experts

### 5.2 Tensor Parallelism (TP)
- **TP degree**: 1 (not used per expert)
- **Optional TP**: TP=2 if expert > GPU memory
- **TP within expert**: Split MLP weights column-wise

### 5.3 Pipeline Parallelism (PP)
- **PP degree**: 4 (one stage per MoE layer)
- **Micro-batches**: 4 micro-batches per batch
- **Schedule**: 1F1B (one forward, one backward) - inference variant

## 6. Implementation Details

### 6.1 Communication Primitives
- **All-to-all**: ncclAllToAll for token routing
- **Broadcast**: ncclBroadcast for expert parameters (if shared)
- **Reduce**: ncclReduce for gradient aggregation (training extension)

### 6.2 Synchronization
- **Barrier**: ncclGroupStart/ncclGroupEnd for collective operations
- **Stream synchronization**: cudaStreamSynchronize between compute/comm streams
- **Token counters**: Atomic operations for load monitoring

### 6.3 Error Handling
- **Expert overflow**: Drop tokens if expert capacity exceeded
- **Network failure**: Retry with exponential backoff
- **Load imbalance**: Dynamic rebalancing every 100 iterations