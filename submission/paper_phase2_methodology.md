# Phase 2: Methodology Extraction

## Methods

### 1. Overview
Our approach focuses on maximizing expert-level parallelism in large-scale Mixture-of-Experts (MoE) models by deploying at most one expert per GPU, and distributing experts across nodes to exploit available compute resources fully. The core idea is to shift the bottleneck from inter-expert contention to network communication, which can be mitigated through careful scheduling, routing, and overlapping of communication and computation.

The method consists of three key components:
1. **Expert Placement Strategy** – Assigning experts across GPUs and nodes.
2. **Routing and Load Balancing** – Ensuring balanced input distribution to experts.
3. **Communication Overlap and Scheduling** – Minimizing the impact of cross-node data transfers.

### 2. Expert Placement Strategy

#### 2.1 Single-Expert-Per-GPU Deployment
In conventional MoE implementations, multiple experts are colocated on a single GPU to reduce cross-node communication. However, this limits the parallelism achievable at the expert level. In contrast, our method deploys at most one expert per GPU:

- For a MoE layer with $E$ experts and a cluster of $G$ GPUs, we ensure that each expert is assigned to a distinct GPU if $E \leq G$.
- If $E > G$, we replicate experts across GPUs in a manner that maximizes the concurrency of independent experts while balancing memory usage.

This approach ensures that each expert can process tokens without contention from other experts on the same device, fully utilizing GPU compute units.

#### 2.2 Cross-Node Distribution
Experts are distributed across nodes to minimize hotspotting on any single node. We use a topology-aware placement strategy that takes into account:

- **Node-to-node bandwidth and latency**
- **GPU memory capacity per node**
- **Expected token routing patterns**

The placement algorithm aims to minimize the maximum number of tokens sent across any single link while maintaining the one-expert-per-GPU principle.

### 3. Routing and Load Balancing

#### 3.1 Gating Mechanism
The routing of tokens to experts is governed by a gating network, as in standard MoE architectures. For each input token, the top-K gating scores determine which experts are activated.

#### 3.2 Token Sharding Across Nodes
Given cross-node expert placement, tokens destined for experts on different nodes must be transferred efficiently. Our approach includes:

1. **Token Batching**: Group tokens by destination expert to reduce the number of network messages.
2. **Asynchronous Routing**: Send token batches asynchronously to overlapping expert computation.
3. **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities to avoid overloading specific experts.

By carefully sharding tokens, we reduce network congestion and ensure that all experts receive a balanced workload, preventing stragglers that could degrade throughput.

### 4. Communication Overlap and Scheduling

#### 4.1 Overlapping Compute and Communication
To mitigate the latency of cross-node token transfers, we interleave expert computation and communication:

- While one batch of tokens is being processed on a GPU, the next batch is simultaneously transferred from other nodes.
- CUDA streams or asynchronous communication libraries (e.g., NCCL or MPI) are leveraged to ensure that data transfer does not block GPU computation.

#### 4.2 Pipeline Scheduling
In multi-layer MoE networks, the scheduling ensures that:

- Token outputs from the previous MoE layer are immediately routed to the next layer's experts.
- Experts in subsequent layers start processing as soon as a partial batch arrives, rather than waiting for the full batch.

This fine-grained pipeline increases throughput and reduces idle time for each expert.

### 5. Scalability Considerations

#### 5.1 Large EP Regime (EP ≥ 16)
Our method is optimized for large EP setups, defined as having 16 or more experts per parallel group. In this regime:

- Network bandwidth becomes the primary limiting factor. We mitigate this by topology-aware routing and token batching.
- The one-expert-per-GPU policy ensures that all GPUs are fully utilized for compute, while communication costs are amortized across many tokens.

#### 5.2 Memory and Model Parallelism Integration
To handle very large models that cannot fit on a single GPU:

- Each expert can be further partitioned using tensor model parallelism (TP) within its GPU if necessary.
- Data parallelism (DP) is applied across replicas of the MoE network, allowing synchronized weight updates while maintaining high expert-level parallelism.

### 6. Summary of Advantages
Our method provides:
1. **Maximized Expert Parallelism:** One expert per GPU ensures minimal contention and high compute efficiency.
2. **Balanced Load Across Nodes:** Topology-aware placement and dynamic gating prevent network bottlenecks.
3. **Scalable Communication Overlap:** Asynchronous token routing allows near-linear scaling for EP ≥ 16.
4. **Compatibility with Large Models:** Integrates seamlessly with TP and DP for models exceeding single-GPU memory.