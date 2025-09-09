# Phase 2: Methodology Extraction

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: Deploy at most one expert per GPU
- **Implementation**: 
  - For E experts and G GPUs: assign each expert to a distinct GPU if E ≤ G
  - If E > G: replicate experts across GPUs to maximize concurrency while balancing memory usage
- **Benefit**: Each expert processes tokens without contention from other experts on the same device

### 1.2 Cross-Node Distribution
- **Topology-aware placement strategy** considers:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum number of tokens sent across any single link while maintaining one-expert-per-GPU principle

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
- Standard MoE gating network determines top-K gating scores for each input token
- Top-K experts are activated per token

### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities to prevent overloading specific experts

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
- **Interleaving Strategy**: While one batch processes on GPU, next batch transfers from other nodes
- **Implementation**: CUDA streams or asynchronous communication libraries (NCCL/MPI)
- **Benefit**: Data transfer does not block GPU computation

### 3.2 Pipeline Scheduling
- **Multi-layer MoE networks**:
  - Token outputs from previous MoE layer immediately routed to next layer's experts
  - Experts in subsequent layers start processing as soon as partial batch arrives
- **Fine-grained pipeline**: Increases throughput and reduces expert idle time

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
- **Definition**: Expert Parallelism degree of 16 or more
- **Network bandwidth**: Primary limiting factor, mitigated by topology-aware routing and token batching
- **One-expert-per-GPU**: Ensures all GPUs fully utilized for compute while communication costs amortized across many tokens

### 4.2 Memory and Model Parallelism Integration
- **Tensor Model Parallelism (TP)**: Applied within single expert's FFN if expert cannot fit on one GPU (optional TP=2)
- **Data Parallelism (DP)**: Applied across replicas of MoE network for synchronized weight updates while maintaining high expert-level parallelism

## 5. Implementation Details

### 5.1 Model Configuration
- **Layers**: 4-layer MoE
- **Experts per layer**: 16
- **Expert type**: MLP
- **Precision**: FP16
- **Token dimension**: 8192
- **MLP hidden size**: 32768
- **Batch size**: 1024 sequences
- **Sequence length**: 10000 tokens
- **MHA**: 16 heads, 512 dimensions per head

### 5.2 Deployment Specifications
- **Proposed Method**: 64 H100 GPUs
- **Per-GPU allocation**: Exactly one expert per GPU
- **Tensor parallelism**: Applied only if single expert's FFN cannot fit (optional TP=2)
- **Pipeline parallelism**: Each MoE layer as micro-stage with overlapped token communication
- **Routing**: Dynamic routing to GPU holding corresponding expert with asynchronous token batches

## 6. Communication Strategy
- **Token transfers**: Asynchronous cross-node transfers
- **Batching**: Group tokens by destination expert
- **Overlap**: Computation and communication interleaved
- **Libraries**: NCCL or MPI for asynchronous communication