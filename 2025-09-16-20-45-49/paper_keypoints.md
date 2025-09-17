# Keypoints of Large-Scale Cross-Node Expert Parallelism for MoE Models

## Core Problem
- Traditional MoE parallelization colocates multiple experts on the same GPU to reduce communication
- This creates computational bottlenecks and limits expert-level parallelism
- Need to maximize compute concurrency in large GPU clusters

## Proposed Solution
- **Large Expert Parallelism**: EP ≥ 16 (at least 16 experts per parallel group)
- **One expert per GPU**: Deploy at most one expert per GPU to maximize parallelism
- **Cross-node distribution**: Distribute experts across nodes to fully utilize distributed resources

## Key Innovations
1. **Expert Placement Strategy**: One expert per GPU, topology-aware placement
2. **Routing and Load Balancing**: Dynamic token routing with load balancing
3. **Communication Overlap**: Asynchronous token routing to overlap compute and communication

## Technical Details
- **Model**: 4-layer MoE, 16 experts per layer, each expert is an MLP
- **Precision**: FP16
- **Batch**: 1024 sequences, 10000 tokens per sequence
- **Dimensions**: Token dim=8192, MHA heads=16, head dim=512, MLP hidden=32768

## Performance Results
- **Baseline (TP=8, PP=2)**: 120,000 TPS, 8.3ms TPOT
- **Proposed Method**: 450,000 TPS, 2.2ms TPOT
- **Improvement**: 3.75× higher throughput, 3.8× lower latency

## Key Benefits
1. Maximized expert parallelism (one expert per GPU)
2. Balanced load across nodes
3. Scalable communication overlap
4. Compatibility with large models (integrates with TP/DP)