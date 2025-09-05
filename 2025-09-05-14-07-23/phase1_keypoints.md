# Phase 1: Key Points of the Paper

## Main Problem Addressed
- Traditional MoE implementations colocate multiple experts on the same GPU to reduce communication overhead
- This creates computational bottlenecks and limits expert-level parallelism
- Need to maximize expert-level parallelism while managing communication costs

## Key Innovation
- **Large-scale cross-node expert parallelism strategy**
- Deploy at most **one expert per GPU** to maximize computational parallelism
- Push Expert Parallelism (EP) to 16 or beyond ("large EP")
- Shift bottleneck from intra-GPU contention to network communication

## Core Contributions
1. **Expert Placement Strategy**: One expert per GPU deployment
2. **Cross-Node Distribution**: Topology-aware placement across nodes
3. **Routing and Load Balancing**: Dynamic token routing with load balancing
4. **Communication Overlap**: Asynchronous token routing overlapping compute and communication
5. **Scalability Framework**: Optimized for EP ≥ 16 regime

## Technical Highlights
- **Model Architecture**: 4-layer MoE with 16 experts per layer, each expert is an MLP
- **Precision**: FP16
- **Batch Configuration**: 1024 sequences × 10000 tokens per sequence
- **Dimensions**: 16 attention heads × 512 dim per head, MLP hidden size 32768

## Performance Achievements
- **Baseline**: 16 H100 GPUs, TP=8, PP=2, 4 experts per GPU → 120,000 TPS, 8.3ms TPOT
- **Proposed**: 64 H100 GPUs, 1 expert per GPU → 450,000 TPS, 2.2ms TPOT
- **Improvement**: 3.75× higher throughput, 3.8× lower latency

## Key Design Principles
1. **Maximize Expert Parallelism**: One expert per GPU ensures minimal contention
2. **Balance Load Across Nodes**: Topology-aware placement prevents bottlenecks
3. **Overlap Communication**: Asynchronous routing enables near-linear scaling
4. **Integrate with Existing Parallelism**: Compatible with TP and DP for large models