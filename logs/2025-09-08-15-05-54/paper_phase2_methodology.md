# Phase 2: Detailed Methodology

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Policy**: At most one expert per GPU
- **Constraint**: For E experts and G GPUs, each expert assigned to distinct GPU if E ≤ G
- **Replication**: If E > G, experts replicated across GPUs to maximize concurrency while balancing memory
- **Benefit**: Each expert processes tokens without contention from other experts on same device

### 1.2 Cross-Node Distribution
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard MoE top-K gating scores determine expert activation per token
- K value not specified in paper (assumed K=2 based on standard MoE practice)

### 2.2 Token Sharding Across Nodes
1. **Token Batching**: Group tokens by destination expert to reduce network messages
2. **Asynchronous Routing**: Send token batches asynchronously to overlap with computation
3. **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Interleaving**: Process current batch while transferring next batch
- **Technology**: CUDA streams or asynchronous communication (NCCL/MPI)
- **Non-blocking**: Data transfer does not block GPU computation

### 3.2 Pipeline Scheduling
- **Micro-staging**: Each MoE layer as separate micro-stage
- **Streaming**: Subsequent layer experts start processing partial batches immediately
- **Throughput**: Fine-grained pipeline increases throughput and reduces idle time

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Expert Parallelism degree ≥ 16
- **Limiting factor**: Network bandwidth
- **Mitigation**: Topology-aware routing and token batching
- **Utilization**: One-expert-per-GPU ensures full GPU utilization

### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Applied within expert if FFN cannot fit on single GPU (optional TP=2)
- **Data Parallelism (DP)**: Applied across MoE network replicas for synchronized weight updates
- **Hierarchy**: EP → TP → DP layering for maximum scalability

## 5. Model Architecture Details
- **Layers**: 4 MoE layers
- **Experts per layer**: 16 (total 64 experts)
- **Expert type**: MLP
- **Token dimension**: 8192
- **MHA configuration**: 16 heads × 512 dim per head = 8192 total
- **MLP hidden dimension**: 32768
- **Precision**: FP16
- **Batch configuration**: 1024 sequences × 10000 tokens per sequence

## 6. Deployment Configurations

### 6.1 Baseline Configuration
- **Parallelism**: TP=8, PP=2
- **GPUs**: 16 H100
- **Per-GPU allocation**:
  - 1/8 tensor-parallel shard for all layers
  - 2 pipeline stages, 8 GPUs per stage
  - 4 experts per GPU (colocated)
- **Processing**: Sequential pipeline flow with shared compute resources

### 6.2 Proposed Configuration
- **Parallelism**: EP=64 (one expert per GPU), optional TP=2 within expert
- **GPUs**: 64 H100
- **Per-GPU allocation**:
  - Exactly one expert per GPU
  - Tensor parallelism only if needed for memory
  - Each MoE layer as micro-stage
- **Routing**: Dynamic routing to GPU holding corresponding expert with asynchronous token batches