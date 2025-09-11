# Keypoints Extraction - Large-Scale Cross-Node Expert Parallelism for MoE Models

## Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Contributions

### 1. Core Innovation
- **Single-expert-per-GPU deployment**: Unlike traditional approaches that place multiple experts on one GPU, this method deploys at most one expert per GPU
- **Large EP regime**: Defines "large EP" as configurations with EP ≥ 16, maximizing expert-level parallelism
- **Cross-node distribution**: Experts are distributed across nodes to minimize hotspotting and fully utilize distributed resources

### 2. Technical Approach
- **Expert Placement Strategy**: Topology-aware placement considering node-to-node bandwidth, GPU memory capacity, and token routing patterns
- **Routing and Load Balancing**: Dynamic token routing with asynchronous routing and load balancing to prevent expert overloading
- **Communication Overlap**: Interleaving expert computation and communication using CUDA streams/NCCL/MPI
- **Pipeline Scheduling**: Fine-grained pipeline ensuring immediate routing between MoE layers

### 3. Scalability Features
- **Large EP Optimization**: Optimized for EP ≥ 16 where network bandwidth becomes the primary limiting factor
- **Integration with Other Parallelisms**: Compatible with tensor model parallelism (TP) and data parallelism (DP)
- **Memory Management**: Each expert can be further partitioned using TP if single expert FFN cannot fit on one GPU

### 4. Experimental Validation
- **Model Configuration**: 4-layer MoE, 16 experts per layer, FP16 precision, 1024 sequences per batch, 10000 tokens per sequence, 8192 token dimension
- **Baseline Comparison**: TP=8, PP=2 configuration with 16 GPUs vs. proposed method with 64 GPUs
- **Performance Gains**: 3.75× higher throughput (450,000 vs 120,000 TPS) and 3.8× lower latency (2.2ms vs 8.3ms TPOT)

### 5. Key Advantages
1. **Maximized Expert Parallelism**: One expert per GPU ensures minimal contention
2. **Balanced Load Across Nodes**: Topology-aware placement prevents network bottlenecks  
3. **Scalable Communication Overlap**: Asynchronous token routing enables near-linear scaling
4. **Large Model Compatibility**: Seamless integration with TP and DP for models exceeding single-GPU memory

## Critical Technical Details
- **Expert Dimensions**: Hidden size of MLP is 32768, token dimension is 8192
- **MHA Configuration**: 16 heads, 512 dimension per head
- **Deployment Constraint**: At most one expert per GPU principle
- **Communication Strategy**: Token batching, asynchronous routing, and pipeline scheduling
- **Performance Metrics**: TPS (Tokens per Second) and TPOT (Time per Output Token)