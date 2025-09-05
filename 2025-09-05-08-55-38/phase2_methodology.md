# Phase 2: Methodology Extraction

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: Each GPU hosts at most one expert
- **Mathematical constraint**: For E experts and G GPUs, ensure E ≤ G for unique assignment
- **Replicate experts**: When E > G, replicate experts across GPUs while maintaining memory balance
- **Benefit**: Eliminates intra-GPU expert contention, maximizes compute unit utilization

### 1.2 Cross-Node Distribution Algorithm
- **Topology-aware placement** considers:
  - Node-to-node bandwidth and latency matrices
  - GPU memory capacity per node (H100: 80GB HBM3)
  - Expected token routing patterns based on gating network
- **Objective**: Minimize max tokens sent across any single network link
- **Constraint**: Maintain one-expert-per-GPU principle

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- **Top-K selection**: K=2 for each token (standard MoE practice)
- **Gating network**: Softmax over expert scores
- **Dynamic adjustment**: Monitor per-expert load and adjust gating probabilities

### 2.2 Token Sharding Process
1. **Token Batching**:
   - Group tokens by destination expert ID
   - Batch size determined by: min(1024 tokens, available network bandwidth)
   - Reduce network messages from O(tokens) to O(experts)

2. **Asynchronous Routing**:
   - Send token batches asynchronously using CUDA streams
   - Overlap with expert computation on destination GPUs
   - Use double-buffering for token transfers

3. **Load Balancing**:
   - Monitor tokens/sec per expert
   - Adjust gating probabilities: P'(expert_i) = P(expert_i) * (1 - α * load_i)
   - α = 0.1 (empirically determined)

## 3. Communication Overlap and Scheduling

### 3.1 Compute-Communication Overlap
- **Mechanism**: While GPU processes current batch, next batch transfers via NCCL
- **Implementation**: 
  - CUDA stream 0: Expert computation
  - CUDA stream 1: Token transfer (send/recv)
  - Event synchronization between streams

### 3.2 Pipeline Scheduling
- **Layer-wise pipeline**: Each MoE layer = one micro-stage
- **Token routing**: Output tokens immediately routed to next layer's experts
- **Partial batch processing**: Start processing when partial tokens arrive
- **Depth**: 4 layers = 4 micro-stages in pipeline

## 4. Memory and Model Parallelism Integration

### 4.1 Tensor Parallelism (TP) for Experts
- **Trigger condition**: When single expert FFN (32768 hidden) doesn't fit in GPU memory
- **TP degree**: TP=2 splits expert across 2 GPUs
- **Implementation**: Column-wise split for linear layers
- **Memory calculation**: 32768 * 512 * 2 bytes (FP16) = 33.5MB per weight matrix

### 4.2 Data Parallelism (DP)
- **DP degree**: Determined by total model replicas needed
- **Synchronization**: All-reduce across DP replicas after each layer
- **Gradient accumulation**: Not applicable (inference-only)

## 5. Scalability Design for EP ≥ 16

### 5.1 Large EP Regime Characteristics
- **Network bottleneck**: Inter-node bandwidth becomes limiting factor
- **Mitigation**: 
  - Token batching reduces messages by 16× (1024 tokens / 64 experts)
  - Topology-aware routing minimizes cross-node traffic
  - H100 NVSwitch provides 900 GB/s intra-node, 400 Gbps InfiniBand inter-node

### 5.2 Resource Utilization
- **GPU utilization**: 100% compute units for expert processing
- **Memory utilization**: ~50GB per GPU (expert weights + activations)
- **Network utilization**: ~75% of available bandwidth with overlap

## 6. Implementation Details

### 6.1 Hardware Configuration
- **GPU**: NVIDIA H100 80GB HBM3
- **Interconnect**: 
  - Intra-node: NVLink 4.0 (900 GB/s)
  - Inter-node: InfiniBand NDR (400 Gbps)
- **Topology**: 8 GPUs per node, 8 nodes for 64 GPU deployment

### 6.2 Software Stack
- **Communication**: NCCL 2.18+, MPI for multi-node
- **CUDA**: CUDA 12.0+
- **Framework**: PyTorch 2.0 with custom MoE kernels
- **Precision**: FP16 for compute, FP32 for master weights (inference)

### 6.3 Model Dimensions Summary
- **Experts per layer**: 16
- **Total layers**: 4
- **Expert hidden size**: 32768
- **MHA heads**: 16
- **Head dimension**: 512
- **Total parameters**: ~1.3B (4 layers × 16 experts × 32768 × 512 × 2)