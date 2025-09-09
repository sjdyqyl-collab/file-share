# Key Points of the Paper: Large-Scale Cross-Node Expert Parallelism for MoE Models

## Core Problem
Traditional MoE parallelization strategies colocate multiple experts on the same GPU to reduce communication, but this creates computational bottlenecks and limits expert-level parallelism as models and clusters grow.

## Proposed Solution
A cross-node expert parallelism method that prioritizes distributing experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond.

## Key Innovation
- **Large EP Definition**: EP ≥ 16 qualifies as "large EP"
- **One-expert-per-GPU policy**: Maximizes compute concurrency by ensuring each expert runs in near isolation
- **Shift in optimization focus**: From reducing communication to maximizing compute concurrency

## Three Key Components
1. **Expert Placement Strategy**: Assigning experts across GPUs and nodes with topology-aware placement
2. **Routing and Load Balancing**: Ensuring balanced input distribution to experts through token batching and asynchronous routing
3. **Communication Overlap and Scheduling**: Minimizing cross-node data transfer impact through compute-communication overlap

## Technical Details
- **Model Architecture**: 4-layer MoE with 16 experts per layer, each expert is an MLP
- **Precision**: FP16
- **Batch Configuration**: 1024 sequences, 10000 tokens per sequence, 8192 token dimension
- **MHA Configuration**: 16 heads, 512 dimension per head
- **MLP Hidden Size**: 32768

## Experimental Results
- **Baseline (TP=8, PP=2)**: 16 H100 GPUs, 4 experts per GPU → 120,000 TPS, 8.3ms TPOT
- **Proposed Method**: 64 H100 GPUs, 1 expert per GPU → 450,000 TPS, 2.2ms TPOT
- **Performance Gain**: ~3.75× higher throughput, ~3.8× lower latency

## Key Advantages
1. Maximized Expert Parallelism: One expert per GPU ensures minimal contention
2. Balanced Load Across Nodes: Topology-aware placement prevents bottlenecks
3. Scalable Communication Overlap: Asynchronous routing enables near-linear scaling
4. Compatibility with Large Models: Integrates with TP and DP for memory-constrained scenarios