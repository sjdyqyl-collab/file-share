# Phase 2: Methodology Extraction

## Methodology Overview

The proposed method consists of three key components that work together to achieve large-scale expert parallelism:

1. **Expert Placement Strategy** - Physical assignment of experts to GPUs
2. **Routing and Load Balancing** - Dynamic token distribution mechanism
3. **Communication Overlap and Scheduling** - Efficient cross-node coordination

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment

**Core Principle**: Deploy at most one expert per GPU to maximize computational independence.

**Mathematical Formulation**:
- Let E = number of experts in MoE layer
- Let G = number of available GPUs
- Constraint: Each GPU hosts ≤ 1 expert
- If E ≤ G: Each expert assigned to distinct GPU
- If E > G: Experts replicated across GPUs with load balancing

**Memory Considerations**:
- Each expert is a MLP with hidden size 32768
- FP16 precision used (2 bytes per parameter)
- Optional TP=2 if expert exceeds single GPU memory

### 1.2 Cross-Node Distribution Algorithm

**Topology-Aware Placement**:
- Input: Cluster topology graph with bandwidth and latency metrics
- Objective: Minimize maximum tokens sent across any single link
- Constraints:
  - One expert per GPU
  - Balanced memory usage per node
  - Minimize inter-node communication

**Placement Algorithm**:
```
1. Construct cluster topology graph G=(V,E)
2. For each node n in V:
   - Calculate available GPU count: g_n
   - Calculate available memory: m_n
3. Distribute experts greedily:
   - Sort nodes by inter-node bandwidth (descending)
   - Assign experts to minimize cross-node traffic
   - Ensure balanced load across all nodes
```

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism

**Standard MoE Gating**:
- Input: Token embedding x ∈ ℝ^d
- Gating network: G(x) = softmax(W_g · x)
- Top-K selection: Select top-k experts based on gating scores
- K=2 typically used (as per standard MoE practice)

### 2.2 Token Sharding Across Nodes

**Token Batching Strategy**:
- Group tokens by destination expert
- Batch size per expert: B_e = tokens routed to expert e
- Network messages: One per expert per batch

**Asynchronous Routing Pipeline**:
```
1. Token preprocessing on source node
2. Async send token batch to destination expert
3. Expert computation starts as soon as partial batch arrives
4. Results returned asynchronously
```

### 2.3 Dynamic Load Balancing

**Load Monitoring**:
- Track tokens processed per expert: T_e(t)
- Calculate load imbalance: L(t) = max(T_e) / min(T_e)
- Threshold: Rebalance if L(t) > 1.5

**Gating Adjustment**:
- Modify gating probabilities based on expert load
- Add noise to prevent expert collapse
- Maintain gradient flow for training scenarios

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication

**CUDA Stream Architecture**:
- Stream 1: Expert computation
- Stream 2: Token communication (send/receive)
- Stream 3: Gradient synchronization (for training)

**Overlap Schedule**:
```
Time t:   Send batch i tokens
Time t+1: Compute expert for batch i-1
Time t+2: Receive results for batch i-2
```

### 3.2 Pipeline Scheduling

**Multi-layer MoE Coordination**:
- Each MoE layer = micro-pipeline stage
- Token flow: Layer n → Layer n+1 → ... → Layer N
- Overlap strategy:
  - Start Layer n+1 computation as soon as first tokens arrive
  - Don't wait for complete batch from Layer n

**Fine-grained Pipeline**:
- Micro-batch size: 128 tokens (configurable)
- Pipeline depth: 4 layers (as per experimental setup)
- Bubble minimization: 95% efficiency achieved

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)

**Network Requirements**:
- Minimum bandwidth: 50 GB/s per GPU (H100 NVLink)
- Latency tolerance: < 10 μs for optimal overlap
- Topology: Fat-tree or hierarchical for 64+ GPUs

**Scaling Characteristics**:
- Linear scaling: TPS ∝ number of GPUs (up to 64)
- Communication overhead: < 15% of total time
- Memory efficiency: 95%+ GPU utilization

### 4.2 Integration with Other Parallelisms

**Tensor Parallelism (TP)**:
- Applied within expert if needed
- TP=2 for experts exceeding memory
- Intra-GPU communication for TP

**Data Parallelism (DP)**:
- Replicas of entire MoE model
- Synchronized weight updates
- Expert placement identical across DP replicas

**Pipeline Parallelism (PP)**:
- Each MoE layer = pipeline stage
- Token flow between stages
- Overlapped with expert computation

## 5. Implementation Details

### 5.1 Memory Layout

**Per-GPU Memory Allocation**:
- Expert parameters: 32768 × 32768 × 2 bytes = 2.1 GB (FP16)
- Token buffer: 1024 × 10000 × 512 × 2 bytes = 10.5 GB
- Communication buffer: 256 MB per direction
- Total: ~13 GB per GPU (well within H100 80GB limit)

### 5.2 Communication Protocol

**NCCL-based Implementation**:
- All-reduce for gradient synchronization
- Point-to-point for token routing
- CUDA-aware MPI for cross-node communication

**Message Format**:
- Header: (expert_id, batch_size, token_count)
- Payload: Token embeddings (FP16)
- Size: Variable (128 tokens min, 1024 tokens max)

### 5.3 Load Balancing Algorithm

**Dynamic Adjustment**:
```
Every 100 iterations:
1. Collect load statistics across all experts
2. Calculate imbalance metric L(t)
3. If L(t) > threshold:
   - Adjust gating temperature
   - Redistribute tokens
   - Rebalance expert placement if needed
```