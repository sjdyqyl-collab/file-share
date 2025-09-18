# Phase 2: Methodology Extraction

## Detailed Methodology

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: Deploy at most one expert per GPU to maximize expert-level parallelism
- **Implementation**: 
  - For MoE layer with E experts and G GPUs: assign each expert to distinct GPU if E ≤ G
  - If E > G: replicate experts across GPUs to maximize concurrency while balancing memory usage
- **Benefit**: Each expert processes tokens without contention from other experts on same device

#### 1.2 Cross-Node Distribution
- **Topology-aware placement strategy** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum number of tokens sent across any single link while maintaining one-expert-per-GPU principle

### 2. Routing and Load Balancing

#### 2.1 Gating Mechanism
- Standard MoE gating network determines top-K experts for each input token
- Uses gating scores to select expert subset

#### 2.2 Token Sharding Across Nodes
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
- **Load Balancing**: 
  - Monitor per-expert load
  - Dynamically adjust gating probabilities to avoid overloading specific experts
  - Prevent stragglers that could degrade throughput

### 3. Communication Overlap and Scheduling

#### 3.1 Overlapping Compute and Communication
- **Interleaving Strategy**:
  - While one batch processes on GPU, next batch transfers simultaneously from other nodes
  - Use CUDA streams or asynchronous communication libraries (NCCL/MPI)
  - Ensure data transfer doesn't block GPU computation

#### 3.2 Pipeline Scheduling
- **Multi-layer MoE networks**:
  - Token outputs from previous MoE layer immediately routed to next layer's experts
  - Experts in subsequent layers start processing partial batches immediately
  - Avoid waiting for full batch completion
- **Benefit**: Fine-grained pipeline increases throughput and reduces idle time

### 4. Scalability Considerations

#### 4.1 Large EP Regime (EP ≥ 16)
- **Network bandwidth** becomes primary limiting factor
- **Mitigation strategies**:
  - Topology-aware routing
  - Token batching
  - One-expert-per-GPU policy ensures full GPU utilization
  - Communication costs amortized across many tokens

#### 4.2 Memory and Model Parallelism Integration
- **Tensor Model Parallelism (TP)**: Applied within single expert's FFN if cannot fit on one GPU (optional TP=2)
- **Data Parallelism (DP)**: Applied across replicas of MoE network for synchronized weight updates
- **Compatibility**: Seamless integration with TP and DP for models exceeding single-GPU memory

### 5. Implementation Details

#### 5.1 Model Configuration
- **Layers**: 4-layer MoE
- **Experts**: 16 experts per layer (64 total experts)
- **Expert Type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16
- **Token Dimension**: 8192
- **Hidden Size of MLP**: 32768
- **MHA Configuration**: 16 heads, 512 dimensions per head

#### 5.2 Deployment Specifications
- **Baseline**: 16 H100 GPUs, TP=8, PP=2
  - 4 experts per GPU
  - Experts colocated on GPUs
  - Pipeline stages span 8 GPUs each
- **Proposed**: 64 H100 GPUs
  - 1 expert per GPU
  - Tensor parallelism optional (TP=2) for large experts
  - Each MoE layer as micro-stage
  - Asynchronous token routing

#### 5.3 Batch Configuration
- **Batch Size**: 1024 sequences per batch
- **Sequence Length**: 10000 tokens per sequence
- **Total Tokens**: 10,240,000 tokens per batch