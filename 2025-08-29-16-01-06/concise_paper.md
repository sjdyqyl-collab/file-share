# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **1. Introduction**
Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per token. However, traditional approaches colocate multiple experts per GPU to reduce communication, creating computational bottlenecks that limit expert-level parallelism as models and clusters scale.

We present a cross-node expert parallelism method that distributes experts such that each GPU hosts at most one expert, pushing EP ≥ 16 to unlock higher concurrent computation. This shifts optimization from communication reduction to compute concurrency maximization, leveraging modern HPC networking capabilities.

## **2. Methods**

### **2.1 Expert Placement Strategy**
- **Single-Expert-Per-GPU**: Deploy at most one expert per GPU
  - For E experts and G GPUs: assign distinct GPU if E ≤ G
  - If E > G: replicate experts while maximizing concurrency
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, memory capacity, and routing patterns

### **2.2 Routing and Load Balancing**
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with computation
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities

### **2.3 Communication Overlap and Scheduling**
- **Compute-Communication Overlap**: Process one batch while transferring next using CUDA streams/NCCL
- **Pipeline Scheduling**: Route outputs immediately to next layer, start processing on partial batches

### **2.4 Large EP Regime (EP ≥ 16)**
- **Definition**: Expert Parallelism degree ≥ 16
- **Characteristics**: Network bandwidth becomes limiting factor, communication costs amortized across tokens
- **Integration**: Combine with tensor parallelism (TP) for memory constraints and data parallelism (DP) for scaling

## **3. Experiments**

### **3.1 Setup**
- **Model**: 4-layer MoE, 16 experts/layer (64 total), MLP experts
- **Precision**: FP16
- **Batch**: 1024 tokens/forward pass
- **Dimensions**: MHA (16 heads × 512), MLP hidden 32,768
- **Hardware**: H100 GPUs, inference-only

### **3.2 Configurations**

#### **Baseline (TP=8, PP=2)**
- **GPUs**: 16 H100
- **Deployment**: 4 experts + 1/8 tensor shard per GPU
- **Parallelism**: TP=8, PP=2 stages, EP=1

#### **Proposed (EP=64)**
- **GPUs**: 64 H100
- **Deployment**: 1 expert per GPU
- **Parallelism**: EP=64, optional TP=2 if memory needed

### **3.3 Results**
| Method | GPUs | TPS | TPOT (ms) |
|--------|------|-----|-----------|
| Baseline | 16 | 120,000 | 8.3 |
| Proposed | 64 | 450,000 | 2.2 |

**Improvements**: 3.75× throughput, 3.8× latency reduction

## **4. Conclusion**
Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU with EP ≥ 16. Through topology-aware placement, asynchronous routing, and computation-communication overlap, we achieve 3.75× higher throughput and 3.8× lower latency compared to traditional approaches, providing a scalable blueprint for high-performance MoE inference.

## **5. Deployment Configuration**

### **Key Parameters**
- **Model**: 4-layer × 16 experts/layer = 64 total experts
- **Expert**: MLP with hidden size 32,768
- **Precision**: FP16
- **Batch**: 1024 tokens
- **Large EP**: EP ≥ 16 (achieved EP=64)

### **Hardware Requirements**
- **Baseline**: 16 H100 GPUs, 2 nodes × 8 GPUs
- **Proposed**: 64 H100 GPUs, 8 nodes × 8 GPUs
- **Network**: 400 Gbps InfiniBand, NVLink within nodes

### **Parallelism Implementation**
- **Baseline**: TP=8, PP=2, 4 experts/GPU
- **Proposed**: EP=64, 1 expert/GPU, optional TP=2 for memory
- **Communication**: NCCL async all-to-all for expert routing