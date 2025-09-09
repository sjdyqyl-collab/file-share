# Phase 1: Key Points Extraction

## **Abstract** (Retained Original)

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Key Points**

### **Problem Statement**
- Traditional MoE parallelization colocates multiple experts per GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- As model and cluster sizes grow, this trade-off becomes suboptimal

### **Proposed Solution**
- Cross-node expert parallelism with at most one expert per GPU
- Large Expert Parallelism (EP ≥ 16) for maximum compute concurrency
- Shifts optimization focus from reducing communication to maximizing compute concurrency

### **Core Components**
1. **Expert Placement Strategy**: One expert per GPU, topology-aware distribution
2. **Routing and Load Balancing**: Dynamic gating with token batching and asynchronous routing
3. **Communication Overlap**: Interleaving computation and communication via CUDA streams

### **Technical Specifications**
- Model: 4-layer MoE, 16 experts per layer, each expert is MLP
- Precision: FP16
- Batch size: 1024 sequences
- Sequence length: 10000 tokens
- Token dimension: 8192
- MHA: 16 heads, 512 dimensions per head
- MLP hidden size: 32768

### **Performance Results**
- Baseline (TP=8, PP=2): 16 GPUs, 120,000 TPS, 8.3ms TPOT
- Proposed: 64 GPUs, 450,000 TPS, 2.2ms TPOT
- Improvement: 3.75× higher throughput, 3.8× lower latency

### **Key Innovation**
- Maximizes expert-level parallelism by dedicating one GPU per expert
- Leverages modern HPC networking (NVLink, InfiniBand, NVSwitch) to handle communication overhead
- Enables near-linear scaling in large EP regimes (EP ≥ 16)