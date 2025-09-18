# Large-Scale Cross-Node Expert Parallelism for MoE Models - Key Points

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Key Points**

### **Problem Statement**
- Traditional MoE parallelization assigns multiple experts to the same GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- As models and clusters grow, this trade-off becomes suboptimal

### **Proposed Solution**
- **Large Expert Parallelism (EP ≥ 16)**: Deploy at most one expert per GPU
- Cross-node expert distribution to fully utilize distributed resources
- Shift bottleneck from intra-GPU contention to network communication
- Leverage modern HPC networking (NVLink, InfiniBand, NVSwitch) to handle communication overhead

### **Method Components**
1. **Expert Placement Strategy**: One expert per GPU, topology-aware placement
2. **Routing and Load Balancing**: Asynchronous token routing with dynamic load balancing
3. **Communication Overlap**: Interleave computation and communication using CUDA streams/NCCL

### **Technical Details**
- **Model Configuration**: 4-layer MoE, 16 experts per layer, MLP experts
- **Precision**: FP16
- **Batch Size**: 1024 sequences
- **Sequence Length**: 10000 tokens
- **Token Dimension**: 8192
- **MHA**: 16 heads, 512 dim per head
- **MLP Hidden Size**: 32768

### **Performance Results**
- **Baseline (TP=8, PP=2)**: 16 GPUs, 4 experts per GPU, 120,000 TPS, 8.3ms TPOT
- **Proposed Method**: 64 GPUs, 1 expert per GPU, 450,000 TPS, 2.2ms TPOT
- **Improvement**: 3.75× higher throughput, 3.8× lower latency

### **Key Innovations**
- Maximized expert parallelism through single-expert-per-GPU deployment
- Asynchronous token routing with computation-communication overlap
- Topology-aware expert placement for balanced network load
- Scalable to large GPU clusters (64+ H100s)

### **Deployment Strategy**
- **Parallelism Configuration**: EP=64 (one expert per GPU), optional TP=2 for large experts
- **GPU Allocation**: Each GPU hosts exactly one expert
- **Communication**: Cross-node token routing with asynchronous transfers
- **Memory Management**: Tensor parallelism within expert if needed, data parallelism across replicas