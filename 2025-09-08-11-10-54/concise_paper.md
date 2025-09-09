# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, traditional MoE parallelization strategies assign multiple experts to the same GPU, creating computational bottlenecks as model and cluster sizes grow. We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond to unlock higher degrees of concurrent computation.

## **Background**

### **Mixture-of-Experts in Large-Scale Models**
MoE models replace transformer FFN layers with multiple "experts," each specializing in different input patterns. A gating mechanism activates only a subset of experts per token, enabling sparse computation and improved efficiency.

### **Parallelism Strategies for MoE**
Standard MoE scaling combines data parallelism (DP), tensor model parallelism (TP), pipeline parallelism (PP), and expert parallelism (EP). Traditional implementations use moderate EP degrees, placing multiple experts per GPU to limit communication. Our approach shifts this paradigm by maximizing EP ≥ 16, leveraging modern HPC networking capabilities.

## **Methods**

### **1. Expert Placement Strategy**

#### **Single-Expert-Per-GPU Deployment**
- Deploy at most one expert per GPU to maximize expert-level parallelism
- For E experts and G GPUs: assign each expert to distinct GPU if E ≤ G
- If E > G: replicate experts to maximize concurrency while balancing memory

#### **Cross-Node Distribution**
- Topology-aware placement considering node-to-node bandwidth/latency, GPU memory, and routing patterns
- Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

### **2. Routing and Load Balancing**

#### **Token Sharding Across Nodes**
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with computation
- **Load Balancing**: Monitor per-expert load and dynamically adjust gating probabilities

### **3. Communication Overlap and Scheduling**

#### **Overlapping Compute and Communication**
- Interleave expert computation and communication using CUDA streams or NCCL/MPI
- While one batch processes, next batch transfers simultaneously

#### **Pipeline Scheduling**
- Token outputs immediately routed to next layer's experts
- Subsequent layers start processing partial batches without waiting for full completion

### **4. Scalability Considerations**
- **Large EP Regime (EP ≥ 16)**: Network bandwidth becomes primary limiting factor, mitigated through topology-aware routing and token batching
- **Memory Integration**: Tensor parallelism within experts if needed, data parallelism across replicas

## **Experiments**

### **Experimental Setup**
- **Model**: 4-layer MoE, 16 experts per layer (64 total experts), MLP experts
- **Precision**: FP16
- **Input**: 1024 sequences × 10000 tokens = 10.24M tokens per batch
- **Dimensions**: Token dim=8192, MLP hidden=32768, MHA=16 heads×512 dim
- **Hardware**: NVIDIA H100 GPUs

### **Parallel Deployment Details**

#### **Baseline (TP=8, PP=2)**
- **GPUs**: 16 H100
- **Configuration**: 4 experts + TP shard per GPU, 2 pipeline stages
- **Limitation**: Experts colocated, causing intra-GPU contention

#### **Proposed Cross-Node Expert Parallelism**
- **GPUs**: 64 H100 (1 expert per GPU)
- **Configuration**: EP=64, optional TP=2 for large experts, each layer as micro-stage
- **Routing**: Asynchronous token routing with computation overlap

### **Results**

| Method | GPUs | Deployment | TPS | TPOT (ms) | Improvement |
|--------|------|------------|-----|-----------|-------------|
| Baseline | 16 | 4 experts/GPU | 120,000 | 8.3 | - |
| **Proposed** | **64** | **1 expert/GPU** | **450,000** | **2.2** | **3.75× TPS, 3.8× latency** |

The proposed method achieves 3.75× higher throughput and 3.8× lower latency by eliminating expert contention and maximizing parallel computation.

## **Conclusion**

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU, achieving significant performance improvements in large-scale MoE inference. The approach provides a scalable blueprint for high-performance MoE deployments, particularly effective in environments with abundant GPU resources.