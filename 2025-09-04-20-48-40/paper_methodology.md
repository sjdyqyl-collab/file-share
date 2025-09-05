# Methodology Extraction - Large-Scale Cross-Node Expert Parallelism

## Core Method Overview
The proposed method maximizes expert-level parallelism in MoE models by deploying at most one expert per GPU across nodes, shifting the bottleneck from intra-GPU contention to network communication.

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Principle
- **Constraint**: Each GPU hosts at most one expert
- **Mathematical Formulation**: For E experts and G GPUs, ensure each expert is assigned to a distinct GPU when E ≤ G
- **Memory Optimization**: When E > G, replicate experts across GPUs to maximize independent expert concurrency while balancing memory usage

### 1.2 Cross-Node Distribution Algorithm
- **Inputs Considered**:
  - Node-to-node bandwidth matrix B[i,j]
  - Node-to-node latency matrix L[i,j]
  - GPU memory capacity per node M[n]
  - Expected token routing probability matrix P[e1,e2]
- **Objective**: Minimize max(tokens_sent_across_any_single_link)
- **Constraint**: Maintain one-expert-per-GPU principle

## 2. Routing and Load Balancing

### 2.1 Gating Network Architecture
- **Top-K Selection**: For each input token, select top-K experts based on gating scores
- **Gating Score Calculation**: softmax(W_gate · x_token)
- **K Value**: Typically K=2 for top-2 gating

### 2.2 Token Sharding Process
1. **Token Batching Phase**:
   - Group tokens by destination expert ID
   - Batch size per expert = ceil(total_tokens / num_experters)
   - Reduce network messages from O(tokens) to O(experts)

2. **Asynchronous Routing**:
   - Non-blocking send operations using NCCL/MPI
   - CUDA streams for overlapping computation and communication
   - Buffer size: 1024 tokens × hidden_dim × sizeof(fp16)

3. **Dynamic Load Balancing**:
   - Monitor queue length per expert: Q[e] = pending_tokens[e]
   - Adjust gating probabilities: P[e] = P[e] × (1 - α × (Q[e] - Q_avg)/Q_avg)
   - α = 0.1 (balancing factor)

## 3. Communication Overlap and Scheduling

### 3.1 Compute-Communication Overlap Strategy
- **Double Buffering**: Maintain two token buffers per GPU
  - Buffer A: Current computation tokens
  - Buffer B: Next batch tokens being received
- **Overlap Timeline**:
  - T0: Start receiving tokens for expert E_i into Buffer B
  - T1: While receiving, process tokens in Buffer A for expert E_i
  - T2: Swap buffers when both operations complete

### 3.2 Pipeline Scheduling for Multi-Layer MoE
- **Layer-wise Pipeline**:
  - Each MoE layer = 1 pipeline stage
  - Token flow: Layer L → routing → Layer L+1
- **Fine-grained Scheduling**:
  - Start processing partial token batches as they arrive
  - No waiting for full batch completion
  - Micro-batch size = 64 tokens (1024/16 experts)

## 4. Scalability Framework

### 4.1 Large EP Regime (EP ≥ 16)
- **Network Requirements**:
  - Minimum bandwidth: 50 GB/s per GPU (InfiniBand HDR)
  - Maximum latency: 5 μs intra-node, 10 μs inter-node
- **Compute Saturation**:
  - Each expert processes: 1024 tokens / 64 experts = 16 tokens per expert
  - Compute time per expert: ~50 μs for MLP forward pass
  - Communication time: ~20 μs for 16 tokens × hidden_dim × fp16

### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism (TP) within Expert**:
  - Applied only when single expert FFN exceeds GPU memory
  - TP degree = 2 (split hidden dimension 32768 → 16384)
  - Communication overhead: 2× all-reduce operations per expert
- **Data Parallelism (DP) across Replicas**:
  - DP degree = total_gpus / (experts_per_layer × layers)
  - Weight synchronization: All-reduce across DP replicas
  - Gradient accumulation: Async updates to maintain pipeline throughput

## 5. Model Architecture Specifications

### 5.1 MoE Layer Configuration
- **Layers**: 4 MoE layers
- **Experts per Layer**: 16 experts
- **Expert Type**: Feed-forward network (MLP)
- **Input Dimension**: 8192 (16 heads × 512 dimensions)
- **Hidden Dimension**: 32768
- **Activation Function**: GeLU
- **Dropout**: 0.1 (training only, not used in inference)

### 5.2 Precision and Memory
- **Data Type**: FP16 (2 bytes per parameter)
- **Model Parameters per Expert**:
  - MLP weights: (8192 × 32768) + (32768 × 8192) = 536,870,912 parameters
  - Total per expert: ~1.07B parameters
  - Memory per expert: 1.07B × 2 bytes = 2.14 GB
- **Activation Memory**:
  - Input activations: 1024 × 8192 × 2 bytes = 16 MB
  - Hidden activations: 1024 × 32768 × 2 bytes = 64 MB

## 6. Implementation Details

### 6.1 Communication Library
- **Primary**: NCCL (NVIDIA Collective Communications Library)
- **Fallback**: MPI for non-NVIDIA environments
- **Primitives Used**:
  - ncclSend/ncclRecv for point-to-point token transfer
  - ncclAllReduce for gradient synchronization
  - CUDA streams for asynchronous operations

### 6.2 Scheduling Algorithm
```
for each layer in [1, 2, 3, 4]:
    for each expert in [1..16]:
        gpu_id = expert_to_gpu_mapping[layer][expert]
        
        # Asynchronous receive tokens
        async_receive_tokens(gpu_id, source_gpus)
        
        # Process current tokens
        process_expert_computation(gpu_id, current_tokens)
        
        # Send results to next layer
        async_send_results(gpu_id, destination_gpus)
```

### 6.3 Load Monitoring
- **Metrics Collected**:
  - Per-expert queue length (tokens)
  - GPU utilization (%)
  - Network bandwidth utilization (GB/s)
  - Token processing latency (μs)
- **Sampling Frequency**: Every 100 iterations
- **Adjustment Threshold**: 10% deviation from average load