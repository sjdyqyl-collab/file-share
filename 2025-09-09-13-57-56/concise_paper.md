# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, traditional MoE parallelization strategies assign multiple experts to the same GPU, creating computational bottlenecks that limit expert-level parallelism as model and cluster sizes grow.

We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU. By pushing Expert Parallelism (EP) to 16 or beyond, we unlock higher degrees of concurrent computation, leveraging modern HPC networking capabilities to sustain high bandwidth and low latency across nodes.

## **Methods**

### **Expert Placement Strategy**
- **Single-Expert-Per-GPU**: Deploy at most one expert per GPU to eliminate intra-GPU contention
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, memory capacity, and routing patterns
- **Assignment Rules**: 
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Experts replicated to maximize concurrency while balancing memory

### **Routing and Load Balancing**
- **Gating**: Top-K gating scores determine expert activation per token
- **Token Sharding**: 
  - Token batching by destination expert to reduce network messages
  - Asynchronous routing to overlap with computation
  - Dynamic load balancing through gating probability adjustment

### **Communication Overlap and Scheduling**
- **Compute-Communication Overlap**: Interleave expert computation with token transfers using CUDA streams/NCCL
- **Pipeline Scheduling**: 
  - Each MoE layer as micro-stage
  - Immediate routing between layers
  - Partial batch processing to reduce idle time

### **Scalability Framework**
- **Large EP Regime**: EP ≥ 16 with network bandwidth as primary limitation
- **Integration**: Compatible with tensor parallelism (within expert) and data parallelism (across replicas)

## **Experiments**

### **Setup**
- **Model**: 4-layer MoE, 16 experts/layer, MLP experts
- **Precision**: FP16
- **Input**: 1024 sequences × 10,000 tokens = 10.24M tokens/batch
- **Dimensions**: 8192 token dim, 32768 MLP hidden, 16×512 MHA
- **Hardware**: H100 GPUs (inference-only)

### **Configurations**
| Method | GPUs | Strategy | Per-GPU Deployment |
|--------|------|----------|-------------------|
| Baseline | 16 | TP=8, PP=2 | 4 experts + TP shard |
| Proposed | 64 | Large EP | 1 expert per GPU |

### **Results**
| Method | TPS (Tokens/s) | TPOT (ms) | Improvement |
|--------|----------------|-----------|-------------|
| Baseline | 120,000 | 8.3 | - |
| Proposed | 450,000 | 2.2 | 3.75× throughput, 3.8× lower latency |

The 4× GPU increase (16→64) yielding 3.75× throughput demonstrates near-linear scaling in the large EP regime.

## **Conclusion**

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU, achieving 3.75× higher throughput and 3.8× lower latency compared to traditional approaches. This provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.