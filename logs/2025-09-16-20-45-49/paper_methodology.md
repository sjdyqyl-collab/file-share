# Methodology: Large-Scale Cross-Node Expert Parallelism

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: At most one expert per GPU
- **Allocation**: 
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Replicate experts across GPUs to maximize concurrency
- **Benefit**: Each expert processes tokens without contention

### 1.2 Cross-Node Distribution
- **Topology-aware placement** considers:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard MoE top-K gating scores determine expert activation
- Each input token routed to top-K experts based on gating scores

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously while overlapping computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Interleaving Strategy**:
  - While batch N processes on GPU, batch N+1 transfers from other nodes
  - Use CUDA streams or NCCL/MPI for asynchronous communication
- **Non-blocking**: Data transfer does not block GPU computation

### 3.2 Pipeline Scheduling
- **Multi-layer coordination**:
  - Token outputs from layer N immediately routed to layer N+1 experts
  - Experts start processing partial batches without waiting for full batch
- **Fine-grained pipeline**: Increases throughput, reduces idle time

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Network bandwidth**: Primary limiting factor
- **Mitigation**: Topology-aware routing and token batching
- **One-expert-per-GPU**: Ensures full GPU utilization

### 4.2 Memory and Model Parallelism Integration
- **Tensor Model Parallelism (TP)**: Partition experts within GPU if needed
- **Data Parallelism (DP)**: Apply across MoE replicas for synchronized updates
- **Compatibility**: Seamless integration with TP and DP for large models

## 5. Implementation Details

### 5.1 Model Architecture
- **Layers**: 4 MoE layers
- **Experts per layer**: 16
- **Expert type**: MLP
- **Precision**: FP16

### 5.2 Input Configuration
- **Batch size**: 1024 sequences
- **Sequence length**: 10000 tokens
- **Token dimension**: 8192
- **MHA**: 16 heads × 512 head dimension = 8192
- **MLP hidden size**: 32768

### 5.3 Deployment Configurations
- **Baseline**: TP=8, PP=2, 16 GPUs total
  - 8 experts per GPU per layer
  - Pipeline stages span 8 GPUs each
- **Proposed**: EP=16, 16 GPUs total
  - 1 expert per GPU per layer
  - Full expert parallelism across all GPUs