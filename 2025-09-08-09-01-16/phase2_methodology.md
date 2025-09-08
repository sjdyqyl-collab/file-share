# Phase 2: Methodology Extraction

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Rule**: At most one expert per GPU
- **Condition**: If E ≤ G (experts ≤ GPUs), each expert assigned to distinct GPU
- **Condition**: If E > G, experts replicated across GPUs maximizing concurrency
- **Benefit**: Each expert processes tokens without contention from other experts

### 1.2 Cross-Node Distribution
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard top-K gating scores determine expert activation per token
- Dynamic gating probabilities to avoid overloading specific experts

### 2.2 Token Sharding Across Nodes
1. **Token Batching**: Group tokens by destination expert to reduce network messages
2. **Asynchronous Routing**: Send token batches asynchronously overlapping expert computation
3. **Load Balancing**: Monitor per-expert load and dynamically adjust gating

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Interleaving**: Process one batch while transferring next batch
- **Technology**: CUDA streams or asynchronous communication (NCCL/MPI)
- **Benefit**: Data transfer doesn't block GPU computation

### 3.2 Pipeline Scheduling
- **Multi-layer MoE**: Token outputs immediately routed to next layer's experts
- **Fine-grained processing**: Subsequent layers start processing partial batches
- **Benefit**: Increases throughput, reduces idle time

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: EP ≥ 16 qualifies as "large EP"
- **Primary limiter**: Network bandwidth (mitigated by topology-aware routing and token batching)
- **Benefit**: One-expert-per-GPU ensures all GPUs utilized while communication amortized

### 4.2 Memory and Model Parallelism Integration
- **Tensor Parallelism (TP)**: Applied within expert's GPU if FFN cannot fit (optional TP=2)
- **Data Parallelism (DP)**: Applied across MoE replicas for synchronized weight updates
- **Compatibility**: Seamless integration with TP and DP for models exceeding single-GPU memory

## 5. Model Architecture Details
- **Layers**: 4-layer MoE
- **Experts per layer**: 16 (64 total experts)
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16
- **Batch size**: 1024 sequences
- **Sequence length**: 10,000 tokens
- **Token dimension**: 8192
- **MHA**: 16 heads, 512 dimension per head
- **MLP hidden size**: 32,768