## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Key Points**

### **Problem Statement**
- Traditional MoE parallelization assigns multiple experts to the same GPU to reduce communication
- This creates computational bottlenecks and limits true expert parallelism
- As model and cluster sizes grow, this trade-off becomes increasingly suboptimal

### **Proposed Solution**
- **Large-scale cross-node expert parallelism** with at most one expert per GPU
- **Large EP regime**: EP ≥ 16 (experts per parallel group)
- **Key principle**: Shift bottleneck from intra-GPU contention to network communication

### **Core Methodology**
1. **Expert Placement**: One expert per GPU, distributed across nodes
2. **Routing & Load Balancing**: Asynchronous token routing with dynamic gating
3. **Communication Overlap**: Interleave computation and communication using CUDA streams
4. **Scalability**: Integrates with TP and DP for memory-constrained scenarios

### **Experimental Results**
- **Setup**: 4-layer MoE, 16 experts/layer, FP16, 1024 tokens/batch
- **Baseline**: TP=8, PP=2, 16 H100s (4 experts/GPU)
- **Proposed**: 64 H100s (1 expert/GPU)
- **Performance**: 3.75× higher throughput (450k vs 120k TPS), 3.8× lower latency (2.2ms vs 8.3ms TPOT)

### **Key Contributions**
- Demonstrates effectiveness of large EP (≥16) in practical deployments
- Shows communication overhead can be mitigated through careful scheduling
- Provides scalable blueprint for high-performance MoE inference
- Validated on H100 clusters with near-linear scaling in large EP regime