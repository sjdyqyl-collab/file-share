# Phase 1: Key Points Extraction

## Main Problem
Traditional MoE implementations colocate multiple experts on the same GPU to reduce communication, creating computational bottlenecks and limiting expert-level parallelism.

## Key Contribution
Proposed a large-scale cross-node expert parallelism strategy that deploys at most one expert per GPU to maximize computational parallelism, specifically designed for Expert Parallelism (EP) ≥ 16.

## Core Methodology
1. **Expert Placement Strategy**: One expert per GPU deployment across nodes
2. **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory
3. **Routing and Load Balancing**: Dynamic token routing with asynchronous batching
4. **Communication Overlap**: Interleaving computation and communication using CUDA streams

## Technical Specifications
- **Model**: 4-layer MoE with 16 experts per layer
- **Expert Type**: MLP with hidden size 32768
- **Token Dimension**: 8192
- **Precision**: FP16
- **Batch Size**: 1024 sequences with 10000 tokens each
- **Hardware**: H100 GPUs

## Experimental Results
- **Baseline (TP=8, PP=2)**: 120,000 TPS, 8.3ms TPOT using 16 GPUs
- **Proposed Method**: 450,000 TPS, 2.2ms TPOT using 64 GPUs
- **Improvement**: 3.75× higher throughput, 3.8× lower latency

## Key Innovation
Shifting optimization focus from reducing communication to maximizing compute concurrency by leveraging modern HPC networking capabilities (NVLink, InfiniBand, NVSwitch).