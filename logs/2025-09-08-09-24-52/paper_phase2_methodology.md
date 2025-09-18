# Large-Scale Cross-Node Expert Parallelism - Methodology

## **1. Overview**
Our approach focuses on maximizing expert-level parallelism in large-scale Mixture-of-Experts (MoE) models by deploying at most one expert per GPU, and distributing experts across nodes to exploit available compute resources fully. The core idea is to shift the computational bottleneck from intra-GPU contention to network communication, which can be mitigated through careful scheduling, routing, and overlapping of communication and computation.

## **2. Expert Placement Strategy**

### **2.1 Single-Expert-Per-GPU Deployment**
- **Principle**: Deploy at most one expert per GPU
- **Implementation**:
  - For MoE layer with E experts and cluster of G GPUs: assign each expert to distinct GPU if E ≤ G
  - If E > G: replicate experts across GPUs to maximize concurrency while balancing memory usage
- **Benefit**: Each expert processes tokens without contention from other experts on same device

### **2.2 Cross-Node Distribution**
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU principle

## **3. Routing and Load Balancing**

### **3.1 Gating Mechanism**
- Standard MoE gating network determines top-K gating scores for each input token
- Subset of experts activated based on gating scores

### **3.2 Token Sharding Across Nodes**
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

## **4. Communication Overlap and Scheduling**

### **4.1 Overlapping Compute and Communication**
- **Mechanism**: Interleave expert computation and communication
  - While one batch processes on GPU, next batch transfers simultaneously from other nodes
  - Use CUDA streams or asynchronous communication libraries (NCCL/MPI)
  - Ensure data transfer doesn't block GPU computation

### **4.2 Pipeline Scheduling**
- **Multi-layer coordination**:
  - Token outputs from previous MoE layer immediately routed to next layer's experts
  - Experts in subsequent layers start processing partial batches as soon as they arrive
  - Avoid waiting for full batch completion

## **5. Scalability Considerations**

### **5.1 Large EP Regime (EP ≥ 16)**
- **Definition**: Large Expert Parallelism when EP ≥ 16
- **Characteristics**:
  - Network bandwidth becomes primary limiting factor
  - Mitigated through topology-aware routing and token batching
  - One-expert-per-GPU ensures all GPUs fully utilized for compute
  - Communication costs amortized across many tokens

### **5.2 Memory and Model Parallelism Integration**
- **Tensor Model Parallelism (TP)**: Applied within expert if single expert's FFN cannot fit on one GPU
- **Data Parallelism (DP)**: Applied across replicas of MoE network for synchronized weight updates
- **Integration**: Maintains high expert-level parallelism while handling very large models

## **6. Implementation Details**

### **6.1 Hardware Requirements**
- **GPUs**: H100-class GPUs with high-bandwidth interconnects
- **Network**: NVLink, InfiniBand, or H100-class NVSwitch fabrics
- **Scale**: 64+ GPUs for large EP deployment

### **6.2 Model Configuration**
- **Layers**: 4 MoE layers
- **Experts per layer**: 16 (baseline) to 64 (large EP)
- **Expert type**: MLP-based feed-forward networks
- **Precision**: FP16
- **Token dimension**: 8192
- **MLP hidden size**: 32768

### **6.3 Parallelism Configuration**
- **Expert Parallelism (EP)**: 64 (one expert per GPU)
- **Tensor Parallelism (TP)**: Optional TP=2 for large experts
- **Pipeline Parallelism (PP)**: Each MoE layer as micro-stage
- **Data Parallelism (DP)**: Across model replicas

## **7. Communication Patterns**
- **Token Distribution**: Asynchronous routing of tokens to destination experts
- **Gradient Synchronization**: All-reduce operations for data parallelism
- **Expert State Transfer**: Minimal as experts remain stationary on assigned GPUs
- **Topology Optimization**: Minimize cross-node traffic through expert placement

## **8. Load Balancing Algorithm**
- **Monitoring**: Track per-expert load and token distribution
- **Adjustment**: Dynamically modify gating probabilities to balance load
- **Feedback**: Use runtime statistics to inform placement decisions
- **Recovery**: Handle straggler experts through dynamic redistribution