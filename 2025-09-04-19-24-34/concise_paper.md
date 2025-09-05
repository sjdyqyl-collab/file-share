# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, scaling MoE models across large GPU clusters introduces significant challenges in expert placement and parallelization. Traditional strategies assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks and limiting expert-level parallelism as model and cluster sizes grow.

We present a cross-node expert parallelism method that prioritizes distributing experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This design shifts the optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities.

## **Methods**

### **1. Expert Placement Strategy**

#### **1.1 Single-Expert-Per-GPU Deployment**
- **Principle**: Each GPU hosts at most one expert
- **Implementation**: For E experts and G GPUs, assign each expert to distinct GPU if E ≤ G; if E > G, replicate experts to maximize concurrency
- **Benefit**: Each expert processes tokens without contention from other experts on same device

#### **1.2 Cross-Node Distribution**
- **Topology-aware placement** considering node-to-node bandwidth/latency, GPU memory capacity, and token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

### **2. Routing and Load Balancing**
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

### **3. Communication Overlap and Scheduling**
- **Overlapping Compute and Communication**: Interleave expert computation and communication using CUDA streams or NCCL/MPI
- **Pipeline Scheduling**: Token outputs immediately routed to next layer's experts; subsequent layers process partial batches without waiting

### **4. Scalability Considerations**
- **Large EP Regime (EP ≥ 16)**: Network bandwidth becomes primary limiting factor, mitigated by topology-aware routing
- **Memory Integration**: Tensor parallelism (TP) within GPU if expert cannot fit; data parallelism (DP) across replicas

## **Experiments**

### **1. Experimental Setup**
- **Model**: 4-layer MoE, 16 experts per layer (MLP)
- **Precision**: FP16
- **Batch size**: 1024 tokens per forward pass
- **MHA**: 16 heads × 512 dimensions per head
- **MLP hidden size**: 32,768
- **Hardware**: H100 GPUs
- **Metrics**: TPS (Tokens per Second), TPOT (Time per Output Token)

### **2. Deployment Configurations**

#### **2.1 Baseline (TP=8, PP=2)**
- **GPUs**: 16 H100
- **Allocation**: 4 experts + 1/8 tensor shard per GPU, 2 pipeline stages
- **Processing**: Sequential pipeline with shared compute resources

#### **2.2 Proposed Cross-Node Expert Parallelism**
- **GPUs**: 64 H100
- **Allocation**: 1 expert per GPU, 64-way expert parallelism
- **Routing**: Dynamic token routing with asynchronous communication overlap

### **3. Results**
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2ms |

**Improvements**: 3.75× higher throughput, 3.8× lower latency

## **Conclusion**

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU, achieving significant performance improvements in large-scale MoE deployments. The approach provides a scalable blueprint for high-performance MoE inference, particularly in environments with abundant GPU resources.

## **Key Technical Specifications**

### **Model Architecture**
- **Layers**: 4 MoE layers
- **Experts per layer**: 16 (64 total experts)
- **Expert type**: MLP
- **Hidden size**: 32,768
- **Precision**: FP16
- **Batch size**: 1024 tokens
- **MHA**: 16 heads × 512 dimensions per head

### **Deployment Requirements**
- **Minimum GPUs**: 64 H100 for full deployment
- **Interconnect**: NVLink, InfiniBand, or H100-class NVSwitch
- **Communication**: NCCL or MPI for asynchronous transfers
- **Scheduling**: CUDA streams for computation-communication overlap
- **Parallelism**: Expert Parallelism (EP) ≥ 16, optional TP=2 for large experts