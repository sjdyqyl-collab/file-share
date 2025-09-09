# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, traditional MoE implementations assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks that limit expert-level parallelism as model and cluster sizes grow.

We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This design shifts the optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities.

## **Methods**

### **Expert Placement Strategy**
- **Single-Expert-Per-GPU**: Deploy at most one expert per GPU to eliminate intra-GPU contention
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, GPU memory, and routing patterns
- **Resource Allocation**: For E experts and G GPUs, ensure distinct GPU assignment when E ≤ G; replicate experts across GPUs when E > G while maximizing concurrency

### **Routing and Load Balancing**
- **Gating Mechanism**: Standard top-K gating scores determine expert activation per token
- **Token Sharding**: Group tokens by destination expert, asynchronous routing, and dynamic load balancing
- **Load Monitoring**: Adjust gating probabilities to prevent expert overloading and ensure balanced workloads

### **Communication Overlap and Scheduling**
- **Compute-Communication Overlap**: Interleave expert computation with cross-node token transfers using CUDA streams or NCCL/MPI
- **Pipeline Scheduling**: Token outputs immediately routed to next layer experts; fine-grained pipeline increases throughput
- **Asynchronous Processing**: Next token batch transfers while current batch computes

### **Scalability Considerations**
- **Large EP Regime**: Optimized for EP ≥ 16 where network bandwidth becomes primary limiting factor
- **Integration**: Compatible with tensor parallelism (TP) within experts and data parallelism (DP) across replicas
- **Memory Handling**: TP applied within single expert if FFN cannot fit on one GPU (optional TP=2)

## **Experiments**

### **Setup**
- **Setting**: Inference-only evaluation
- **Model**: 4-layer MoE, 16 experts per layer (MLP experts)
- **Precision**: FP16
- **Input**: 1024 sequences × 10,000 tokens = 10.24M tokens per batch
- **Dimensions**: Token dimension 8192, MLP hidden size 32768, 16 attention heads × 512 dimensions
- **Hardware**: H100 GPUs in HPC cluster

### **Configurations**
- **Baseline**: 16 GPUs with TP=8, PP=2, 4 experts per GPU
- **Proposed**: 64 GPUs with 1 expert per GPU, EP=64, each MoE layer as micro-stage

### **Results**
| Method | GPUs | TPS (Tokens/s) | TPOT (ms) |
|--------|------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 120,000 | 8.3 |
| Proposed | 64 | 450,000 | 2.2 |

**Improvements**: 3.75× higher throughput, 3.8× lower latency

## **Conclusion**

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying at most one expert per GPU. In an inference-only setting, the approach achieves 3.75× higher throughput and 3.8× lower latency compared to traditional methods, providing a scalable blueprint for high-performance MoE inference in HPC environments.