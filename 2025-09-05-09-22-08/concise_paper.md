# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, traditional MoE parallelization strategies assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks that limit expert-level parallelism as model and cluster sizes grow.

We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This design shifts optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities.

## **Background**

### **Mixture-of-Experts in Large-Scale Models**
MoE models replace transformer FFN layers with multiple "experts" (specialized MLPs), with a gating mechanism activating only a subset per token, enabling sparse computation and improved efficiency.

### **Parallelism Strategies for MoE**
Standard implementations combine data parallelism (DP), tensor model parallelism (TP), pipeline parallelism (PP), and expert parallelism (EP), typically placing multiple experts per GPU to limit communication. However, as network interconnects advance (NVLink, InfiniBand, NVSwitch), communication cost becomes less dominant than compute concurrency gains.

### **Large Expert Parallelism (Large EP)**
We define *large EP* as configurations where EP ≥ 16. In this regime, distributing one expert per GPU minimizes resource contention and maximizes expert-level parallel execution, with network bandwidth becoming the primary limiting factor.

## **Methods**

### **1. Expert Placement Strategy**

#### **1.1 Single-Expert-Per-GPU Deployment**
- **Principle**: At most one expert per GPU
- **Constraint**: For E experts and G GPUs, assign each expert to distinct GPU if E ≤ G
- **Replication**: If E > G, replicate experts to maximize independent concurrency while balancing memory
- **Benefit**: Eliminates intra-GPU contention, fully utilizes GPU compute units

#### **1.2 Cross-Node Distribution**
Topology-aware placement considering:
- Node-to-node bandwidth and latency
- GPU memory capacity per node
- Expected token routing patterns
- Objective: Minimize maximum tokens sent across any single link

### **2. Routing and Load Balancing**

#### **2.1 Token Sharding Strategy**
1. **Token Batching**: Group tokens by destination expert to reduce network messages
2. **Asynchronous Routing**: Send token batches asynchronously to overlap expert computation
3. **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

### **3. Communication Overlap and Scheduling**

#### **3.1 Compute-Communication Overlap**
- **Mechanism**: Interleave expert computation with cross-node token transfers
- **Implementation**: CUDA streams with NCCL/MPI for asynchronous operations
- **Pipeline**: While batch N processes, batch N+1 transfers simultaneously

#### **3.2 Pipeline Scheduling**
- Each MoE layer as a micro-stage
- Token outputs immediately routed to next layer's experts
- Experts start processing partial batches without waiting for full completion

### **4. Memory and Model Parallelism Integration**
- **Tensor Parallelism (TP)**: Optional TP=2 within expert if single expert's FFN cannot fit on GPU
- **Data Parallelism (DP)**: Synchronized weight updates across replicas
- **Pipeline Parallelism**: Each layer as a micro-stage with overlapped communication

## **Experiments**

### **1. Experimental Setup**
- **Model**: 4-layer MoE, 16 experts per layer, each expert is an MLP
- **Precision**: FP16
- **Batch Size**: 1024 tokens per forward pass
- **MHA Configuration**: 16 heads, 512 dimensions per head
- **MLP Hidden Size**: 32,768
- **Hardware**: H100 GPUs in inference-only setting

### **2. Parallel Deployment Configurations**

#### **2.1 Baseline (TP=8, PP=2)**
- **GPUs**: 16 H100
- **Per-GPU**: 4 experts + 1/8 tensor-parallel shard
- **Pipeline**: 2 stages, 8 GPUs each
- **Experts**: Colocated, 4 per GPU

#### **2.2 Proposed Cross-Node Expert Parallelism**
- **GPUs**: 64 H100
- **Per-GPU**: Exactly one expert
- **Expert Parallelism**: 64-way (EP=64)
- **Pipeline**: 4 micro-stages (one per layer)
- **Tensor Parallelism**: Optional TP=2 per expert if needed

### **3. Results**

| Method | GPUs | Per-GPU Deployment | TPS | TPOT |
|--------|------|-------------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 ms |
| Proposed Cross-Node EP | 64 | 1 expert per GPU | 450,000 | 2.2 ms |

**Performance Improvements:**
- **3.75× higher throughput** (450k vs 120k TPS)
- **3.8× lower latency** (2.2ms vs 8.3ms TPOT)
- **93.75% scaling efficiency** (3.75× performance with 4× GPUs)

### **4. Key Findings**
- Single expert per GPU eliminates intra-GPU contention
- Asynchronous token routing minimizes waiting time
- Near-linear scaling achieved in large EP regime (EP ≥ 16)
- Network bandwidth effectively managed through topology-aware routing

## **Conclusion**

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU. This approach achieved 3.75× higher throughput and 3.8× lower latency compared to baseline configurations by fully utilizing 64 GPUs and enabling large Expert Parallelism (EP ≥ 64). The method provides a scalable blueprint for high-performance MoE inference in GPU-rich environments like H100 clusters.

## **Deployment Configuration**

See `deployment_config.json` for complete technical specifications including:
- Parallel strategy parameters for both baseline and proposed methods
- Detailed module configurations with dimensions
- Device mapping strategies
- Communication backend settings
- Batch configuration parameters