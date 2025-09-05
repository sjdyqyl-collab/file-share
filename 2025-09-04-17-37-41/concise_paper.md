# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, traditional MoE parallelization strategies assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks that limit expert-level parallelism as model and cluster sizes grow.

We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU. By pushing Expert Parallelism (EP) to 16 or beyond, we unlock higher degrees of concurrent computation, allowing each expert to run in near isolation. This design leverages modern HPC networking capabilities to shift the optimization focus from reducing communication to maximizing compute concurrency.

## **Methods**

### **1. Expert Placement Strategy**

**Single-Expert-Per-GPU Deployment**:
- At most one expert per GPU constraint
- If E ≤ G: Each expert assigned to distinct GPU
- If E > G: Experts replicated while maximizing concurrency
- Eliminates intra-GPU expert contention

**Cross-Node Distribution**:
- Topology-aware placement considering node bandwidth, latency, GPU memory, and routing patterns
- Minimizes maximum tokens sent across any single link
- Maintains one-expert-per-GPU principle

### **2. Routing and Load Balancing**

**Token Sharding Across Nodes**:
- Token batching by destination expert to reduce network messages
- Asynchronous routing to overlap with computation
- Dynamic gating probability adjustment for load balancing

**Gating Mechanism**:
- Standard top-K gating scores determine expert activation
- Monitoring per-expert load with dynamic adjustment
- Prevention of expert overload

### **3. Communication Overlap and Scheduling**

**Overlapping Compute and Communication**:
- Interleave expert computation with token transfers
- CUDA streams/NCCL for asynchronous communication
- Data transfer does not block GPU computation

**Pipeline Scheduling**:
- Token outputs immediately route to next layer experts
- Subsequent layers start processing partial batches
- Fine-grained pipeline reduces expert idle time

### **4. Scalability Considerations**

**Large EP Regime (EP ≥ 16)**:
- Network bandwidth becomes primary limiting factor
- Mitigated through topology-aware routing and token batching
- One-expert-per-GPU ensures full GPU utilization

**Integration with Other Parallelism**:
- Tensor parallelism within expert if memory exceeds GPU capacity
- Data parallelism across MoE replicas
- Maintains high expert-level parallelism

## **Experiments**

### **1. Experimental Setup**

**Model Configuration**:
- 4-layer MoE with 16 experts per layer (64 total experts)
- Each expert: MLP with hidden dimension 32768
- Precision: FP16
- Batch size: 1024 tokens per forward pass
- Attention: 16 heads × 512 dimensions = 8192 total

**Hardware**: H100 GPUs
- Baseline: 16 GPUs
- Proposed: 64 GPUs

**Metrics**: TPS (Tokens/Second), TPOT (Time per Output Token in ms)

### **2. Deployment Configurations**

**Baseline (TP=8, PP=2)**:
- 16 H100 GPUs total
- Tensor parallelism: 8-way
- Pipeline parallelism: 2 stages (8 GPUs each)
- 4 experts per GPU (colocated)
- Sequential token processing through stages

**Proposed Cross-Node Expert Parallelism**:
- 64 H100 GPUs total
- Expert parallelism: 64-way (one expert per GPU)
- Optional tensor parallelism: TP=2 if needed for memory
- Pipeline: Each MoE layer as micro-stage
- Asynchronous token routing with computation overlap

### **3. Results**

| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 ms |
| Proposed | 64 | 1 expert per GPU | 450,000 | 2.2 ms |

**Performance Improvements**:
- **3.75× higher throughput** (450k vs 120k tokens/sec)
- **3.8× lower latency** (2.2ms vs 8.3ms TPOT)
- Full GPU utilization without expert contention
- Near-linear scaling demonstrated

### **4. Key Findings**

- **One expert per GPU** enables maximal expert-level parallelism
- **Asynchronous token routing** ensures minimal waiting across nodes
- **Large EP regime (EP=64)** achieves near-linear scaling
- **Communication-computation overlap** mitigates network latency

## **Conclusion**

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU. By shifting from intra-GPU contention to optimized communication, we achieve 3.75× higher throughput and 3.8× lower latency compared to traditional approaches. This provides a scalable blueprint for high-performance MoE inference in large GPU clusters, particularly effective in the large EP regime (EP ≥ 16).

## **Critical Parameters Summary**

- **Expert Parallelism**: EP ≥ 16 (large EP regime)
- **Total Experts**: 64 (4 layers × 16 experts)
- **GPU Allocation**: 1 expert per GPU
- **Precision**: FP16
- **Batch Size**: 1024 tokens
- **Hidden Dimension**: 32768 (MLP)
- **Attention**: 16 heads × 512 dim = 8192 total
- **Performance**: 450k TPS, 2.2ms TPOT with 64 GPUs