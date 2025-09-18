# Phase 1: Keypoints Extraction

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Points Summary

### Core Problem
- Traditional MoE approaches colocate multiple experts on same GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- Need to shift from communication optimization to compute concurrency maximization

### Proposed Solution
- **Large Expert Parallelism (EP ≥ 16)**: Deploy at most one expert per GPU
- Cross-node expert distribution to maximize computational parallelism
- Focus on compute concurrency over communication reduction

### Technical Innovations
1. **Single-Expert-Per-GPU Deployment**: Each GPU hosts exactly one expert
2. **Cross-Node Distribution**: Topology-aware expert placement across nodes
3. **Communication Overlap**: Asynchronous token routing to overlap compute and communication
4. **Load Balancing**: Dynamic gating to prevent expert overloading

### Performance Gains
- **3.75× higher throughput** (450K vs 120K tokens/second)
- **3.8× lower latency** (2.2ms vs 8.3ms per token)
- Near-linear scaling in large EP regime

### Experimental Validation
- **Model**: 4-layer MoE, 16 experts per layer
- **Deployment**: 64 H100 GPUs (1 expert per GPU) vs 16 H100 baseline
- **Settings**: FP16, batch=1024 sequences, seq_len=10000, token_dim=8192
- **Results**: Demonstrated significant performance improvement

### Scalability Features
- Compatible with tensor parallelism for large experts
- Integrates with data parallelism for training
- Designed for HPC environments with high-bandwidth interconnects

### Key Insight
The paradigm shift from "minimize communication" to "maximize compute concurrency" enables better scaling in large GPU clusters with modern interconnects.