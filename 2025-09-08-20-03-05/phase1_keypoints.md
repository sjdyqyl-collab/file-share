# Phase 1: Key Points Extraction

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Points

### 1. Core Innovation
- **One expert per GPU deployment**: Unlike traditional methods that place multiple experts on a single GPU, this approach assigns at most one expert per GPU
- **Large EP regime**: Expert Parallelism (EP) ≥ 16, maximizing expert-level parallel execution
- **Cross-node distribution**: Experts are distributed across nodes to fully utilize available compute resources

### 2. Problem Addressed
- Traditional MoE parallelization creates computational bottlenecks by colocating multiple experts per GPU
- Limits degree of true expert parallelism as model and cluster sizes grow
- Trade-off between communication cost and compute concurrency becomes suboptimal

### 3. Technical Approach
- **Expert Placement Strategy**: Topology-aware placement considering node-to-node bandwidth, GPU memory, and routing patterns
- **Routing and Load Balancing**: Dynamic gating with token batching and asynchronous routing
- **Communication Overlap**: Interleaving expert computation with cross-node token transfers using CUDA streams or NCCL/MPI

### 4. Experimental Validation
- **Model**: 4-layer MoE with 16 experts per layer (64 total experts)
- **Configuration**: FP16 precision, 1024 sequences per batch, 10000 tokens per sequence, 8192 token dimension
- **Hardware**: H100 GPUs
- **Results**: 3.75× higher throughput (450K vs 120K tokens/s) and 3.8× lower latency (2.2ms vs 8.3ms)

### 5. Deployment Comparison
- **Baseline**: 16 GPUs with TP=8, PP=2, 4 experts per GPU
- **Proposed**: 64 GPUs with 1 expert per GPU, EP=64
- **Key difference**: Eliminates intra-GPU contention and enables maximal expert-level parallelism

### 6. Scalability Features
- Compatible with tensor model parallelism (TP) for experts exceeding single-GPU memory
- Integrates with data parallelism (DP) for synchronized weight updates
- Near-linear scaling in large EP regime (EP ≥ 16)

### 7. Critical Dimensions
- Token dimension: 8192
- Hidden size of MLP: 32768
- Number of heads: 16, head dimension: 512
- Batch size: 1024 sequences
- Sequence length: 10000 tokens