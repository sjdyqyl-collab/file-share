# Phase 1: Key Points Extraction

## Main Contribution
- **Large-scale cross-node expert parallelism strategy** for Mixture-of-Experts (MoE) models
- **One expert per GPU deployment** to maximize computational parallelism
- **Expert Parallelism (EP) ≥ 16** defined as "large EP" regime
- **3.75× higher throughput** and **3.8× lower latency** compared to baseline

## Key Technical Innovations
1. **Expert Placement Strategy**: At most one expert per GPU to minimize contention
2. **Cross-Node Distribution**: Topology-aware placement across nodes
3. **Routing and Load Balancing**: Dynamic token routing with asynchronous communication
4. **Communication Overlap**: Interleaving computation and communication using CUDA streams

## Problem Addressed
- Traditional MoE implementations colocate multiple experts per GPU, creating computational bottlenecks
- Limited expert-level parallelism in conventional approaches
- Need to scale MoE models across large GPU clusters efficiently

## Solution Approach
- Shift bottleneck from intra-GPU contention to network communication
- Leverage modern HPC networking (NVLink, InfiniBand, NVSwitch)
- Maximize compute concurrency through expert distribution
- Overlap communication with computation to mitigate latency

## Experimental Validation
- **Model**: 4-layer MoE, 16 experts per layer, MLP experts
- **Precision**: FP16
- **Batch size**: 1024 tokens
- **Hardware**: H100 GPUs
- **Results**: 450,000 TPS vs 120,000 TPS baseline

## Architecture Details
- **MHA**: 16 heads, 512 dimensions per head
- **MLP hidden size**: 32768
- **Baseline**: TP=8, PP=2, 16 GPUs total, 4 experts per GPU
- **Proposed**: 64 GPUs total, 1 expert per GPU, EP=64