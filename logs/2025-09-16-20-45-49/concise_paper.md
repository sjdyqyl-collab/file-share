# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, traditional MoE parallelization strategies assign multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks that limit expert-level parallelism.

Our approach prioritizes distributing experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This shifts the optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities.

## **Methods**

### **1. Expert Placement Strategy**
- **Single-Expert-Per-GPU**: Deploy at most one expert per GPU
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory
- **Allocation**: If E ≤ G, each expert assigned to distinct GPU; if E > G, replicate experts to maximize concurrency

### **2. Routing and Load Balancing**
- **Gating Mechanism**: Standard top-K gating scores determine expert activation
- **Token Sharding**: Group tokens by destination expert, asynchronous routing, dynamic load balancing
- **Load Balancing**: Monitor per-expert load and adjust gating probabilities

### **3. Communication Overlap and Scheduling**
- **Compute-Communication Overlap**: Interleave expert computation with token transfers using CUDA streams/NCCL
- **Pipeline Scheduling**: Immediate routing between layers, processing partial batches
- **Asynchronous Operations**: Ensure data transfer doesn't block GPU computation

### **4. Scalability Considerations**
- **Large EP Regime**: EP ≥ 16 optimizes for network bandwidth as primary limiting factor
- **Integration**: Compatible with tensor model parallelism (TP) and data parallelism (DP) for large models

## **Experiments**

### **Experimental Setup**
- **Model**: 4-layer MoE, 16 experts per layer (MLP), FP16 precision
- **Input**: 1024 sequences, 10000 tokens/sequence, 8192 token dimension
- **Hardware**: 16 H100 GPUs, inference-only setting
- **Metrics**: TPS (Tokens/second), TPOT (Time per output token in ms)

### **Deployment Configurations**

#### **Baseline (TP=8, PP=2)**
- **GPUs**: 16 total
- **Allocation**: 8 experts per GPU per layer, colocated with tensor-parallel shards
- **Processing**: Sequential pipeline flow with shared compute resources

#### **Proposed Cross-Node Expert Parallelism**
- **GPUs**: 16 total
- **Allocation**: 1 expert per GPU per layer, full expert parallelism
- **Routing**: Dynamic token routing with asynchronous batch sending

### **Results**
| Method | GPUs | Per-GPU Deployment | TPS | TPOT |
|--------|------|-------------------|-----|------|
| Baseline | 16 | 8 experts + TP shard | 120,000 | 8.3ms |
| Proposed | 16 | 1 expert per GPU | 450,000 | 2.2ms |

**Improvements**: 3.75× higher throughput, 3.8× lower latency

## **Conclusion**

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU. This approach achieved 3.75× higher throughput and 3.8× lower latency compared to baseline configurations, demonstrating the effectiveness of large EP (≥16) in high-performance MoE deployments. The method provides a scalable blueprint for future high-performance MoE inference in GPU-rich environments.

## **Model Deployment Configuration**

See `deployment_configuration.json` for complete JSON configuration including:
- Parallel strategy parameters for both baseline and proposed methods
- Detailed module divisions and expert placements
- Device-to-GPU mapping for 16 H100 GPUs across 2 nodes
- Communication patterns and routing configurations