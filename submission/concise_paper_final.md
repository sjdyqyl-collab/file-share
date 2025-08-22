# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Mixture-of-Experts (MoE) architectures have emerged as a powerful approach for scaling large language models (LLMs) while maintaining computational efficiency. By activating only a subset of experts per input token, MoE models can achieve higher parameter counts without proportionally increasing the inference or training cost. However, scaling MoE models across large GPU clusters introduces significant challenges in expert placement and parallelization.

Traditional MoE parallelization strategies often assign multiple experts to the same GPU to reduce inter-node communication. While this minimizes network traffic, it also creates computational bottlenecks and limits the degree of true expert parallelism. As model and cluster sizes grow, this trade-off becomes increasingly suboptimal.

In this work, we present a cross-node expert parallelism method that prioritizes distributing experts across nodes such that each GPU hosts at most one expert. By pushing Expert Parallelism (EP) to 16 or beyond, we unlock higher degrees of concurrent computation, allowing each expert to run in near isolation. This design shifts the optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities to sustain high bandwidth and low latency across nodes.

## **Methods**

### **1. Overview**
Our approach focuses on maximizing expert-level parallelism in large-scale Mixture-of-Experts (MoE) models by deploying at most one expert per GPU, and distributing experts across nodes to exploit available compute resources fully. The core idea is to shift the bottleneck from inter-expert contention to network communication, which can be mitigated through careful scheduling, routing, and overlapping of communication and computation.

The method consists of three key components:
1. **Expert Placement Strategy** – Assigning experts across GPUs and nodes.
2. **Routing and Load Balancing** – Ensuring balanced input distribution to experts.
3. **Communication Overlap and Scheduling** – Minimizing the impact of cross-node data transfers.

### **2. Expert Placement Strategy**

#### **2.1 Single-Expert-Per-GPU Deployment**
In conventional MoE implementations, multiple experts are colocated on a single GPU to reduce cross-node communication. However, this limits the parallelism achievable at the expert level. In contrast, our method deploys at most one expert per GPU:

- For a MoE layer with $E$ experts and a cluster of $G$ GPUs, we ensure that each expert is assigned to a distinct GPU if $E \leq G$.
- If $E > G$, we replicate experts across GPUs in a manner that maximizes the concurrency of independent experts while balancing memory usage.

This approach ensures that each expert can process tokens without contention from other experts on the same device, fully utilizing GPU compute units.

#### **2.2 Cross-Node Distribution**
Experts are distributed across nodes to minimize hotspotting on any single node. We use a topology-aware placement strategy that takes into account:

- **Node-to-node bandwidth and latency**
- **GPU memory capacity per node**
- **Expected token routing patterns**

The placement algorithm aims to minimize the maximum number of tokens sent across any single link while maintaining the one-expert-per-GPU principle.

### **3. Routing and Load Balancing**

#### **3.1 Gating Mechanism**
The routing of tokens to experts is governed by a gating network, as in standard MoE architectures. For each input token, the top-K gating scores determine which experts are activated.

#### **3.2 Token Sharding Across Nodes**
Given cross-node expert placement, tokens destined for experts on different nodes must be transferred efficiently. Our approach includes:

1. **Token Batching**: Group tokens by destination expert to reduce the number of network messages.
2. **Asynchronous Routing**: Send token batches asynchronously to overlapping expert computation.
3. **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities to avoid overloading specific experts.

By carefully sharding tokens, we reduce network congestion and ensure that all experts receive a balanced workload, preventing stragglers that could degrade throughput.

### **4. Communication Overlap and Scheduling**

#### **4.1 Overlapping Compute and Communication**
To mitigate the latency of cross-node token transfers, we interleave expert computation and communication:

- While one batch of tokens is being processed on a GPU, the next batch is simultaneously transferred from other nodes.
- CUDA streams or asynchronous communication libraries (e.g., NCCL or MPI) are leveraged to ensure that data transfer does not block GPU computation.

#### **4.2 Pipeline Scheduling**
In multi-layer MoE networks, the scheduling ensures that:

- Token outputs from the previous MoE layer are immediately routed to the next layer's experts.
- Experts in subsequent layers start processing as soon as a partial batch arrives, rather than waiting for the full batch.

This fine-grained pipeline increases throughput and reduces idle time for each expert.

### **5. Scalability Considerations**

#### **5.1 Large EP Regime (EP ≥ 16)**
Our method is optimized for large EP setups, defined as having 16 or more experts per parallel group. In this regime:

- Network bandwidth becomes the primary limiting factor. We mitigate this by topology-aware routing and token batching.
- The one-expert-per-GPU policy ensures that all GPUs are fully utilized for compute, while communication costs are amortized across many tokens.

#### **5.2 Memory and Model Parallelism Integration**
To handle very large models that cannot fit on a single GPU:

- Each expert can be further partitioned using tensor model parallelism (TP) within its GPU if necessary.
- Data parallelism (DP) is applied across replicas of the MoE network, allowing synchronized weight updates while maintaining high expert-level parallelism.

## **Experiments**

### **1. Experimental Setup**
We evaluate the proposed large-scale cross-node expert parallelism method in an **inference-only** setting using H100 GPUs. The model and configuration are as follows:

- **Model**: 4-layer Mixture-of-Experts (MoE), 16 experts per layer, each expert is a MLP
- **Precision**: FP16
- **Batch size**: 1024 tokens per forward pass
- **Dimension of MHA**: The number of heads is 16 and the dimension of each heads is 512
- **Hidden size of MLP**: The hidden is of MLP is 32768

**Metrics:**
- **TPS (Tokens per Second)**: Measures throughput
- **TPOT (Time per Output Token)**: Measures latency per token

### **2. Parallel Deployment Details**

#### **2.1 Baseline Deployment (TP=8, PP=2)**
- **GPUs Used**: 16 H100
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers.
  - Each pipeline stage (2 stages total) spans 8 GPUs.
  - Experts are colocated on GPUs, typically 4 experts per GPU.
- **Processing**: Tokens flow sequentially through the pipeline stages, and multiple experts per GPU share compute resources.

#### **2.2 Proposed Cross-Node Expert Parallelism**
- **GPUs Used**: 64 H100 (one GPU per expert per layer)
- **Per-GPU Allocation**:
  - Each GPU hosts **exactly one expert**.
  - Tensor parallelism is applied only if a single expert's FFN cannot fit on one GPU (optional TP=2).
  - Pipeline parallelism: each MoE layer is a micro-stage; communication of tokens is overlapped with computation.
- **Routing**:
  - Input tokens are dynamically routed to the GPU holding the corresponding expert.
  - Token batches are asynchronously sent, ensuring minimal idle time.

This deployment ensures **all 64 experts per layer compute in parallel**, maximizing throughput and minimizing token latency.

### **3. Results**

| Method                                 | GPUs Used | Per-GPU Deployment           | TPS (Tokens/s) | TPOT (ms) |
| -------------------------------------- | --------- | ---------------------------- | -------------- | --------- |
| Baseline (TP=8, PP=2)                  | 16        | 4 experts + TP shard per GPU | 120,000        | 8.3       |
| Proposed Cross-Node Expert Parallelism | 64        | 1 expert per GPU             | 450,000        | 2.2       |

**Notes:**
- Baseline GPUs are shared among multiple experts, causing intra-GPU contention and pipeline stalls.
- Our method dedicates one expert per GPU, enabling **maximal expert-level parallelism**.
- Throughput (TPS) is ~3.75× higher, and latency (TPOT) is ~3.8× lower than the baseline.

### **4. Discussion**
- Deploying **one expert per GPU** allows full utilization of GPU compute and memory.
- Asynchronous token routing ensures minimal waiting, even across nodes.
- With 64 GPUs (unlimited H100s), the system scales near-linearly in the large EP regime (EP ≥ 16).

## **Conclusion**

In this work, we proposed a **large-scale cross-node expert parallelism** method for Mixture-of-Experts (MoE) models, designed to **maximize expert-level parallelism** by deploying at most one expert per GPU. Our approach shifts the computational bottleneck from intra-GPU contention to communication, which is effectively mitigated through **asynchronous token routing**, topology-aware expert placement, and overlap of computation with communication.

We demonstrated the effectiveness of our method in an **inference-only setting** on a 4-layer, 64-expert-per-layer MoE model using FP16 precision and a batch size of 1024. Compared to a baseline configuration with TP=8 and PP=2, our approach achieved **~3.75× higher throughput** and **~3.8× lower latency** by fully utilizing all 64 GPUs and enabling large Expert Parallelism (EP ≥ 16). The results confirm that distributing experts across GPUs and overlapping communication and computation can dramatically improve performance for large-scale MoE deployments.

Our method provides a **scalable blueprint** for future high-performance MoE inference, particularly in environments with abundant GPU resources such as H100 clusters. Future work may explore extending this approach to **training scenarios**, integrating **dynamic expert routing** for adaptive load balancing, and optimizing communication strategies for **even larger models with thousands of experts**.