# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models (Concise Version)

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, traditional MoE parallelization strategies assign multiple experts to the same GPU to reduce communication, creating computational bottlenecks as model and cluster sizes grow.

We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU. By pushing Expert Parallelism (EP) to 16 or beyond, we unlock higher degrees of concurrent computation, allowing each expert to run in near isolation. This design shifts optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities.

## **Methods**

### **Expert Placement Strategy**
- **Single-Expert-Per-GPU**: Deploy at most one expert per GPU, ensuring no intra-GPU contention
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory capacity
- **Expert Assignment**: For E experts and G GPUs, assign distinct GPUs when E ≤ G; replicate experts when E > G

### **Routing and Load Balancing**
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously overlapping expert computation
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities to prevent overloading

### **Communication Overlap and Scheduling**
- **Compute-Communication Overlap**: Process current batch while transferring next batch
- **Pipeline Scheduling**: Token outputs immediately routed to next layer's experts
- **Asynchronous Operations**: CUDA streams/NCCL for non-blocking data transfer

### **Scalability Considerations**
- **Large EP Regime**: Optimized for EP ≥ 16 where network bandwidth is limiting factor
- **Integration**: Compatible with tensor parallelism (TP) and data parallelism (DP) for large models
- **Memory Management**: Each expert can use TP within GPU if single expert exceeds memory

## **Experiments**

### **Setup**
- **Model**: 4-layer MoE, 16 experts per layer (64 total experts)
- **Architecture**: Each expert is MLP
- **Dimensions**: Token dimension 8192, MLP hidden size 32768, 16 MHA heads × 512
- **Precision**: FP16
- **Batch**: 1024 sequences × 10000 tokens per sequence

### **Configurations**
| Method | GPUs | Parallelism | Deployment | TPS | TPOT |
|--------|------|-------------|------------|-----|------|
| Baseline | 16 H100 | TP=8, PP=2 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed | 64 H100 | EP=64 | 1 expert per GPU | 450,000 | 2.2ms |

### **Results**
- **Throughput**: 3.75× improvement (450,000 vs 120,000 TPS)
- **Latency**: 3.8× reduction (2.2ms vs 8.3ms TPOT)
- **Scalability**: Near-linear scaling in large EP regime (EP ≥ 16)

## **Conclusion**

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU. The approach achieved 3.75× higher throughput and 3.8× lower latency compared to traditional methods. This provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.

## **Key Dimensions**
- 4 MoE layers
- 16 experts per layer (64 total)
- Token dimension: 8192
- MLP hidden size: 32768
- Batch: 1024 sequences × 10000 tokens
- Precision: FP16
- 16 MHA heads × 512 dimensions = 8192 total