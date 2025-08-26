## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Traditional MoE parallelization assigns multiple experts to the same GPU to reduce communication, creating computational bottlenecks that limit true expert parallelism. Our method distributes experts across nodes with at most one expert per GPU, prioritizing compute concurrency over communication reduction. This design leverages modern HPC networking to achieve high bandwidth and low latency across nodes.

## **Methods**

### **Expert Placement Strategy**
- **Single-Expert-Per-GPU**: Deploy at most one expert per GPU
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory
- **Allocation Rule**: For E experts and G GPUs, assign each expert to distinct GPU if E ≤ G; replicate experts if E > G

### **Routing and Load Balancing**
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously while overlapping expert computation
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities

### **Communication Overlap and Scheduling**
- **Compute-Communication Overlap**: Process current batch while transferring next batch using CUDA streams
- **Pipeline Scheduling**: Each MoE layer as micro-stage, start processing partial batches immediately
- **Implementation**: NCCL/MPI for communication, asynchronous operations

### **Scalability Considerations**
- **Large EP Regime**: EP ≥ 16 for maximum parallelism
- **Integration**: Compatible with tensor parallelism (TP) and data parallelism (DP)
- **Memory Management**: TP=2 only if single expert exceeds GPU memory

## **Experiments**

### **Setup**
- **Model**: 4-layer MoE, 16 experts/layer, FP16 precision
- **Batch Size**: 1024 tokens per forward pass
- **Hardware**: H100 GPUs, inference-only
- **Metrics**: TPS (Tokens/Second), TPOT (Time per Output Token)

### **Configurations**
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts/GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert/GPU | 450,000 | 2.2ms |

### **Results**
- **3.75× throughput improvement** (450k vs 120k TPS)
- **3.8× latency reduction** (2.2ms vs 8.3ms TPOT)
- **Near-linear scaling** achieved with 64 GPUs in large EP regime

## **Conclusion**

Our large-scale cross-node expert parallelism method achieves significant performance improvements by maximizing expert-level parallelism through one-expert-per-GPU deployment. The approach successfully shifts bottlenecks from computational contention to manageable network communication, validated on H100 clusters with substantial throughput and latency gains.

## **Deployment DAG Requirements**

### **Node Configuration**
- **GPU Count**: 64 H100s minimum for full deployment
- **Network**: High-bandwidth interconnects (NVLink/InfiniBand)
- **Memory**: Each GPU hosts exactly one expert

### **Parallelism Configuration**
- **Expert Parallelism (EP)**: 64 (one expert per GPU)
- **Tensor Parallelism (TP)**: 1 (or 2 if expert exceeds memory)
- **Pipeline Parallelism (PP)**: Each MoE layer as micro-stage
- **Data Parallelism (DP)**: Applied across replicas

### **Communication Pattern**
1. **Input Token Distribution**: Async routing to expert locations
2. **Expert Computation**: Parallel processing on assigned GPUs
3. **Output Collection**: Async gathering from all experts
4. **Pipeline Flow**: Immediate forwarding to next layer experts

### **Resource Mapping**
- **Layer 1**: Experts 0-15 → GPUs 0-15
- **Layer 2**: Experts 16-31 → GPUs 16-31
- **Layer 3**: Experts 32-47 → GPUs 32-47
- **Layer 4**: Experts 48-63 → GPUs 48-63