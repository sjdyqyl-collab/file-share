# Phase 2: Methodology Extraction

## 1. Overview

Our approach focuses on maximizing expert-level parallelism in large-scale Mixture-of-Experts (MoE) models by deploying at most one expert per GPU, and distributing experts across nodes to exploit available compute resources fully. The core idea is to shift the computational bottleneck from inter-expert contention to network communication, which is effectively mitigated through careful scheduling, routing, and overlapping of communication and computation.

The method consists of three key components:
1. **Expert Placement Strategy** – Assigning experts across GPUs and nodes
2. **Routing and Load Balancing** – Ensuring balanced input distribution to experts
3. **Communication Overlap and Scheduling** – Minimizing the impact of cross-node data transfers

## 2. Expert Placement Strategy

### 2.1 Single-Expert-Per-GPU Deployment

**Core Principle**: Deploy at most one expert per GPU to eliminate intra-GPU contention.

**Implementation Details**:
- For a MoE layer with E experts and a cluster of G GPUs, ensure each expert is assigned to a distinct GPU if E ≤ G
- If E > G, replicate experts across GPUs to maximize concurrency while balancing memory usage
- Each expert runs in complete isolation on its dedicated GPU

**Memory Considerations**:
- Each GPU hosts exactly one expert's parameters
- No expert parameter sharing within a GPU
- Tensor parallelism (TP) can be applied within an expert if it exceeds single-GPU memory

### 2.2 Cross-Node Distribution

**Topology-Aware Placement**:
- **Node-to-node bandwidth**: Prioritize high-bandwidth connections for frequently communicating experts
- **GPU memory capacity**: Balance memory usage across nodes
- **Expected token routing patterns**: Place frequently co-activated experts on nearby nodes

**Placement Algorithm**:
1. Calculate communication matrix based on expected expert activation patterns
2. Apply graph partitioning to minimize cross-node traffic
3. Ensure load balancing across all GPUs
4. Validate memory constraints per node

## 3. Routing and Load Balancing

### 3.1 Gating Mechanism

**Standard MoE Gating**:
- Top-K gating scores determine expert activation (typically K=2)
- Gating network outputs probability distribution over all experts
- Tokens routed to top-K experts based on gating scores

### 3.2 Token Sharding Across Nodes

**Token Batching Strategy**:
```
Input: Batch of tokens B = [t1, t2, ..., tn]
Process:
1. For each token, determine destination experts via gating
2. Group tokens by destination expert/node
3. Create message batches for each destination
4. Send asynchronously while computation proceeds
```

**Asynchronous Routing Pipeline**:
1. **Pre-processing**: Compute gating scores for entire batch
2. **Sharding**: Group tokens by destination expert
3. **Communication**: Send token batches to destination GPUs
4. **Computation**: Process tokens as they arrive
5. **Gather**: Collect results from all experts

### 3.3 Load Balancing

**Dynamic Gating Adjustment**:
- Monitor per-expert load in real-time
- Adjust gating probabilities to prevent expert overloading
- Implement token dropping threshold for severely overloaded experts
- Use exponential moving average for load estimation

## 4. Communication Overlap and Scheduling

### 4.1 Overlapping Compute and Communication

**CUDA Stream Architecture**:
- **Compute Stream**: Handles expert computation
- **Communication Stream**: Handles token transfers
- **Synchronization**: CUDA events for stream coordination

**Overlap Strategy**:
```
Timeline:
Time 0-1: Send tokens for batch n+1 while computing batch n
Time 1-2: Compute batch n+1 while receiving results for batch n
Time 2-3: Overlap send/receive for consecutive batches
```

### 4.2 Pipeline Scheduling

**Multi-layer MoE Pipeline**:
- Each MoE layer treated as a micro-stage
- Token outputs immediately routed to next layer's experts
- Fine-grained pipeline with token-level granularity

**Scheduling Algorithm**:
1. Token arrives at layer l
2. Compute gating for layer l+1 while processing layer l
3. Send tokens to layer l+1 experts as soon as layer l completes
4. Overlap computation across layers

## 5. Scalability Considerations

### 5.1 Large EP Regime (EP ≥ 16)

**Network Optimization**:
- **Bandwidth requirement**: Sustained 100+ Gbps per GPU for token transfers
- **Latency hiding**: Overlap computation with communication
- **Topology utilization**: Leverage NVLink, InfiniBand, NVSwitch fabrics

**Scaling Characteristics**:
- Linear scaling up to 64 experts (64 GPUs)
- Communication overhead grows sub-linearly with expert count
- Compute efficiency remains constant regardless of expert count

### 5.2 Memory and Model Parallelism Integration

**Tensor Parallelism within Expert**:
- Apply TP only when single expert exceeds GPU memory
- TP=2 splits expert FFN across 2 GPUs
- Maintain one-expert-per-GPU principle at expert level

**Data Parallelism Integration**:
- DP applied across complete MoE model replicas
- Each replica uses large EP strategy internally
- Synchronized weight updates across replicas

## 6. Implementation Details

### 6.1 Memory Layout

**Per-GPU Memory Allocation**:
- Expert parameters: 8192 × 32768 × 2 bytes (FP16) = 512 MB
- Token buffer: 1024 × 10000 × 8192 × 2 bytes = 160 GB (shared across batch)
- Communication buffer: 8192 × 1024 × 2 bytes = 16 MB per message

### 6.2 Communication Patterns

**Token Transfer Protocol**:
- Message size: Variable (depends on token count)
- Frequency: Per-batch
- Pattern: All-to-all with topology optimization
- Bandwidth: 450 GB/s aggregate across cluster

### 6.3 Load Balancing Metrics

**Monitoring Variables**:
- Expert utilization: tokens processed per expert per second
- Network traffic: bytes transferred per GPU per second
- Queue depth: pending tokens per expert
- Drop rate: tokens dropped due to overload

**Adjustment Parameters**:
- Gating temperature: controls expert selection randomness
- Load threshold: maximum tokens per expert
- Re-route probability: chance of redirecting overloaded tokens