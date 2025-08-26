## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Methods**

### **1. Overview**
Our approach focuses on maximizing expert-level parallelism in large-scale Mixture-of-Experts (MoE) models by deploying at most one expert per GPU, and distributing experts across nodes to exploit available compute resources fully. The core idea is to shift the bottleneck from inter-expert contention to network communication, which can be mitigated through careful scheduling, routing, and overlapping of communication and computation.

The method consists of three key components:
1. **Expert Placement Strategy** – Assigning experts across GPUs and nodes
2. **Routing and Load Balancing** – Ensuring balanced input distribution to experts
3. **Communication Overlap and Scheduling** – Minimizing the impact of cross-node data transfers

### **2. Expert Placement Strategy**

#### **2.1 Single-Expert-Per-GPU Deployment**
- **Policy**: At most one expert per GPU
- **Allocation**: For E experts and G GPUs, assign each expert to distinct GPU if E ≤ G
- **Replication**: If E > G, replicate experts to maximize independent expert concurrency
- **Benefit**: Eliminates intra-GPU contention between experts

#### **2.2 Cross-Node Distribution**
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

### **3. Routing and Load Balancing**

#### **3.1 Gating Mechanism**
- Standard top-K gating scores determine expert activation per token

#### **3.2 Token Sharding Across Nodes**
1. **Token Batching**: Group tokens by destination expert to reduce network messages
2. **Asynchronous Routing**: Send token batches asynchronously while overlapping expert computation
3. **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities

### **4. Communication Overlap and Scheduling**

#### **4.1 Overlapping Compute and Communication**
- **Interleaving**: Process current batch while transferring next batch
- **Implementation**: CUDA streams or asynchronous communication libraries (NCCL/MPI)
- **Benefit**: Data transfer does not block GPU computation

#### **4.2 Pipeline Scheduling**
- **Micro-stages**: Each MoE layer as a micro-stage
- **Immediate routing**: Token outputs routed immediately to next layer's experts
- **Partial batch processing**: Start processing as soon as partial batch arrives

### **5. Scalability Considerations**

#### **5.1 Large EP Regime (EP ≥ 16)**
- **Definition**: Expert Parallelism degree ≥ 16
- **Network optimization**: Topology-aware routing and token batching mitigate bandwidth limitations
- **Compute utilization**: One-expert-per-GPU ensures full GPU utilization

#### **5.2 Memory and Model Parallelism Integration**
- **Tensor Parallelism (TP)**: Applied within expert if FFN exceeds single-GPU memory
- **Data Parallelism (DP)**: Applied across MoE network replicas for synchronized weight updates
- **Compatibility**: Seamlessly integrates with existing parallelism strategies

### **6. Implementation Details**
- **Precision**: FP16
- **Communication**: NCCL/MPI for cross-node transfers
- **Scheduling**: CUDA streams for asynchronous operations
- **Load monitoring**: Runtime per-expert load tracking
- **Dynamic adjustment**: Runtime gating probability modification

### **7. Summary of Advantages**
1. **Maximized Expert Parallelism**: One expert per GPU ensures minimal contention
2. **Balanced Load Across Nodes**: Topology-aware placement prevents bottlenecks
3. **Scalable Communication Overlap**: Asynchronous routing enables near-linear scaling for EP ≥ 16
4. **Large Model Compatibility**: Integrates with TP and DP for memory-constrained scenarios