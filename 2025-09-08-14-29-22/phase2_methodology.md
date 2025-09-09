# Phase 2: Methodology Extraction

## Expert Placement Strategy

### Single-Expert-Per-GPU Deployment
- Deploy at most one expert per GPU
- For E experts and G GPUs: assign each expert to distinct GPU if E ≤ G
- If E > G: replicate experts across GPUs to maximize concurrency while balancing memory
- Ensures each expert processes tokens without contention from other experts on same device

### Cross-Node Distribution
- Topology-aware placement considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- Minimizes maximum number of tokens sent across any single link
- Maintains one-expert-per-GPU principle

## Routing and Load Balancing

### Gating Mechanism
- Standard MoE gating network determines top-K experts for each token
- Top-K gating scores determine which experts are activated

### Token Sharding Across Nodes
1. **Token Batching**: Group tokens by destination expert to reduce network messages
2. **Asynchronous Routing**: Send token batches asynchronously overlapping expert computation
3. **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities to prevent overloading

## Communication Overlap and Scheduling

### Overlapping Compute and Communication
- Interleave expert computation and communication
- Process current batch while transferring next batch from other nodes
- Use CUDA streams or asynchronous communication (NCCL/MPI) to prevent blocking

### Pipeline Scheduling
- Token outputs from previous MoE layer immediately routed to next layer's experts
- Experts in subsequent layers start processing as soon as partial batch arrives
- Fine-grained pipeline increases throughput and reduces idle time

## Scalability Considerations

### Large EP Regime (EP ≥ 16)
- Network bandwidth becomes primary limiting factor
- Mitigated through topology-aware routing and token batching
- One-expert-per-GPU policy ensures full GPU utilization while amortizing communication costs

### Memory and Model Parallelism Integration
- Each expert can be partitioned using tensor model parallelism (TP) within GPU if needed
- Data parallelism (DP) applied across replicas of MoE network for synchronized weight updates
- Maintains high expert-level parallelism

## Method Summary
1. **Maximized Expert Parallelism**: One expert per GPU ensures minimal contention and high compute efficiency
2. **Balanced Load Across Nodes**: Topology-aware placement and dynamic gating prevent network bottlenecks
3. **Scalable Communication Overlap**: Asynchronous token routing enables near-linear scaling for EP ≥ 16
4. **Compatibility with Large Models**: Integrates seamlessly with TP and DP for models exceeding single-GPU memory