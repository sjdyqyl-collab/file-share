# Phase 1: Key Points Extraction

## Abstract (Retained Original)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Points

### 1. Core Problem Addressed
- Traditional MoE parallelization places multiple experts on same GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- Challenge: scaling MoE models across large GPU clusters with efficient expert placement

### 2. Proposed Solution
- **Large-scale cross-node expert parallelism**
- Deploy **at most one expert per GPU**
- Push Expert Parallelism (EP) to 16 or beyond ("large EP" regime)
- Shift optimization focus from communication reduction to compute concurrency maximization

### 3. Key Innovations
- **Single-expert-per-GPU deployment**: Eliminates intra-GPU expert contention
- **Cross-node distribution**: Topology-aware expert placement across nodes
- **Asynchronous token routing**: Overlaps communication with computation
- **Pipeline scheduling**: Fine-grained inter-layer processing

### 4. Technical Components
1. **Expert Placement Strategy**: One expert per GPU, topology-aware distribution
2. **Routing and Load Balancing**: Dynamic gating with token batching and load balancing
3. **Communication Overlap**: CUDA streams/NCCL for async communication
4. **Scalability**: Optimized for EP ≥ 16 regime

### 5. Performance Gains
- **3.75× higher throughput** (450k vs 120k tokens/sec)
- **3.8× lower latency** (2.2ms vs 8.3ms TPOT)
- Near-linear scaling with 64 GPUs (EP=64)
- Full GPU utilization without expert contention

### 6. Experimental Validation
- **Model**: 4-layer MoE, 16 experts/layer, 64 total experts
- **Hardware**: 64 H100 GPUs vs 16 H100 baseline
- **Precision**: FP16
- **Batch size**: 1024 tokens
- **Setting**: Inference-only

### 7. Critical Dimensions and Parameters
- **EP degree**: ≥16 (large EP regime)
- **Expert count**: 64 experts per layer (4 layers × 16 experts)
- **Hidden size**: 32768 (MLP hidden dimension)
- **Attention heads**: 16 heads × 512 dimensions = 8192 total
- **GPU allocation**: 1 expert per GPU (64 GPUs total)