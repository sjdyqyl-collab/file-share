# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, traditional MoE parallelization strategies colocate multiple experts on the same GPU to reduce communication, creating computational bottlenecks that limit expert-level parallelism as models and clusters grow.

We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, achieving EP ≥ 16. This design shifts the optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities for high bandwidth and low latency.

## **Methods**

### **1. Expert Placement Strategy**
- **Single-Expert-Per-GPU**: Deploy at most one expert per GPU to eliminate intra-GPU contention
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, memory capacity, and routing patterns
- **Scalability**: When E > G, replicate experts to maximize independent expert concurrency

### **2. Routing and Load Balancing**
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with computation
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

### **3. Communication Overlap and Scheduling**
- **Compute-Communication Overlap**: Interleave expert computation with cross-node token transfers using CUDA streams or NCCL/MPI
- **Pipeline Scheduling**: Route token outputs immediately to next layer's experts, starting processing on partial batches
- **Large EP Optimization**: Optimized for EP ≥ 16 where network bandwidth is the primary limiting factor

### **4. Integration with Existing Parallelism**
- **Tensor Parallelism (TP)**: Applied within single expert if FFN exceeds GPU memory (optional TP=2)
- **Data Parallelism (DP)**: Applied across MoE network replicas for synchronized weight updates
- **Pipeline Parallelism (PP)**: Each MoE layer as micro-stage with overlapped communication

## **Experiments**

### **Setup**
- **Model**: 4-layer MoE, 16 experts per layer (64 total experts), each expert is MLP
- **Precision**: FP16
- **Input**: 1024 sequences × 10,000 tokens per sequence, 8,192 token dimension
- **Attention**: 16 heads × 512 dimensions per head
- **MLP**: 32,768 hidden size
- **Hardware**: H100 GPUs, inference-only

### **Configurations**

| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| **Proposed** | 64 | 1 expert per GPU | **450,000** | **2.2ms** |

### **Results**
- **Throughput**: 3.75× improvement (450,000 vs 120,000 tokens/second)
- **Latency**: 3.8× reduction (2.2ms vs 8.3ms per token)
- **Scaling**: Near-linear scaling with 4× more GPUs yielding 3.75× throughput
- **Efficiency**: 93.75% scaling efficiency in large EP regime

## **Conclusion**

Our large-scale cross-node expert parallelism method achieves significant performance improvements by dedicating one expert per GPU and leveraging asynchronous token routing. With EP ≥ 16, we demonstrate 3.75× higher throughput and 3.8× lower latency compared to traditional approaches. This provides a scalable blueprint for high-performance MoE inference in GPU-rich environments, with potential extensions to training and dynamic routing for future work.