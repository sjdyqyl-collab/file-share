# Phase 1: Key Points Extraction

## Core Problem
Traditional MoE implementations place multiple experts on the same GPU to reduce communication overhead, but this creates computational bottlenecks and limits expert-level parallelism as model and cluster sizes grow.

## Proposed Solution
Large-scale cross-node expert parallelism strategy for MoE models that:
- Deploys at most one expert per GPU
- Uses Expert Parallelism (EP) ≥ 16 ("large EP")
- Maximizes computational parallelism by fully exploiting distributed resources
- Reduces expert-level contention and improves throughput

## Key Technical Components
1. **Expert Placement Strategy**: One expert per GPU, cross-node distribution with topology-aware placement
2. **Routing and Load Balancing**: Token batching, asynchronous routing, dynamic load balancing
3. **Communication Overlap and Scheduling**: Overlapping compute and communication, pipeline scheduling

## Experimental Setup
- Model: 4-layer MoE, 16 experts per layer, each expert is MLP
- Precision: FP16
- Batch: 1024 sequences × 10000 tokens per sequence
- Token dimension: 8192
- MHA: 16 heads × 512 dimensions per head
- MLP hidden size: 32768

## Results
- Baseline (TP=8, PP=2, 16 GPUs): 120,000 TPS, 8.3ms TPOT
- Proposed (64 GPUs, 1 expert per GPU): 450,000 TPS, 2.2ms TPOT
- Improvement: 3.75× higher throughput, 3.8× lower latency

## Key Dimensions to Retain
- 4 MoE layers
- 16 experts per layer (64 total experts)
- Token dimension: 8192
- MLP hidden size: 32768
- Batch size: 1024 sequences
- Sequence length: 10000 tokens
- Precision: FP16
- 16 heads × 512 = 8192 MHA dimensions