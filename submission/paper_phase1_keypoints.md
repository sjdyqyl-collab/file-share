# Phase 1: Key Points Extraction

## Core Problem Addressed
- Traditional MoE implementations colocate multiple experts on the same GPU to reduce communication overhead
- This creates computational bottlenecks and limits expert-level parallelism
- The paper proposes distributing at most one expert per GPU to maximize computational parallelism

## Key Innovation
- Large-scale cross-node expert parallelism strategy for MoE models
- Definition: "Large EP" = Expert Parallelism (EP) ≥ 16
- One expert per GPU deployment to eliminate intra-GPU contention
- Shifts optimization focus from communication reduction to compute concurrency maximization

## Main Contributions
1. **Expert Placement Strategy**: Deploy at most one expert per GPU across nodes
2. **Routing and Load Balancing**: Dynamic token routing with balanced workload distribution
3. **Communication Overlap**: Asynchronous token routing to overlap computation and communication
4. **Scalability**: Near-linear scaling for EP ≥ 16 in HPC environments

## Technical Approach
- Topology-aware expert placement considering bandwidth, latency, and memory
- Token batching and asynchronous routing to reduce network messages
- Pipeline scheduling for multi-layer MoE networks
- Integration with tensor parallelism (TP) and data parallelism (DP) for large models

## Performance Results
- **Model**: 4-layer MoE, 16 experts per layer, FP16 precision
- **Baseline**: TP=8, PP=2 with 16 H100 GPUs (4 experts per GPU)
- **Proposed**: 64 H100 GPUs (1 expert per GPU)
- **Improvement**: 3.75× higher throughput (450k vs 120k TPS), 3.8× lower latency (2.2ms vs 8.3ms TPOT)

## Key Insights
- Network bandwidth becomes primary limiting factor in large EP regime
- One-expert-per-GPU ensures full GPU utilization while communication costs are amortized
- Modern HPC networking (NVLink, InfiniBand, NVSwitch) enables this approach
- Method provides scalable blueprint for future high-performance MoE inference

## Original Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.