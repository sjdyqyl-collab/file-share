# Phase One: Keypoints Extraction

## **Abstract**
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Key Points**

### **Problem Statement**
- Traditional MoE implementations colocate multiple experts per GPU to reduce communication overhead
- This creates computational bottlenecks and limits expert-level parallelism as models and clusters scale
- Need to shift from minimizing communication to maximizing compute concurrency

### **Proposed Solution**
- **Large-scale cross-node expert parallelism** with EP ≥ 16
- **One expert per GPU deployment** to eliminate intra-GPU contention
- **Topology-aware placement** considering bandwidth, latency, and memory capacity
- **Asynchronous token routing** with communication-computation overlap

### **Technical Innovation**
- **Expert Placement Strategy**: Distribute experts across nodes with at most one expert per GPU
- **Routing & Load Balancing**: Token batching, asynchronous routing, dynamic gating adjustment
- **Communication Overlap**: Interleave expert computation with cross-node token transfers
- **Scalability**: Optimized for EP ≥ 16 regime with near-linear scaling

### **Experimental Validation**
- **Model**: 4-layer MoE, 16 experts/layer, MLP experts, FP16 precision
- **Setup**: 64 H100 GPUs (vs 16 for baseline)
- **Results**: 3.75× higher throughput (450k vs 120k TPS), 3.8× lower latency (2.2ms vs 8.3ms TPOT)
- **Baseline**: TP=8, PP=2 with 4 experts per GPU
- **Proposed**: 1 expert per GPU with full expert parallelism

### **Key Dimensions & Parameters**
- **EP (Expert Parallelism)**: Must be ≥ 16 for "large EP" regime
- **Model Architecture**: 4 layers × 16 experts/layer = 64 total experts
- **Expert Structure**: MLP with hidden size 32768
- **Precision**: FP16
- **Batch Size**: 1024 tokens per forward pass
- **MHA Configuration**: 16 heads × 512 dimensions per head

### **Deployment Requirements**
- **GPU Count**: 64 H100 GPUs (one per expert per layer)
- **Memory**: Sufficient for single expert per GPU
- **Network**: High-bandwidth, low-latency interconnect (NVLink, InfiniBand)
- **Parallelism**: EP=64 (maximum), optional TP=2 if expert doesn't fit single GPU

### **Performance Gains**
- **Throughput**: 450,000 tokens/second (vs 120,000 baseline)
- **Latency**: 2.2ms per output token (vs 8.3ms baseline)
- **Resource Utilization**: Full GPU compute utilization with no expert contention
- **Scalability**: Near-linear scaling in large EP regime