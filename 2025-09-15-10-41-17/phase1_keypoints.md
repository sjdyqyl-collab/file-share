# Keypoints of Ring Attention with Sequence Parallelism Paper

## Main Contribution
- Novel parallelization strategy combining **Ring Attention** with **Sequence Parallelism** for Multi-Head Attention in large-scale transformers
- Addresses communication bottlenecks and memory constraints in distributed transformer inference

## Core Innovations
1. **Ring Attention**: Uses ring topology to distribute attention computation across devices with sequential peer-to-peer exchanges instead of all-to-all communication
2. **Sequence Parallelism**: Splits input sequences across workers to reduce memory footprint by factor of P (number of devices)
3. **Combined Approach**: Integrates both techniques to minimize communication overhead while reducing memory usage

## Technical Benefits
- **Communication Efficiency**: Reduces peak bandwidth requirements from all-to-all to ring-based communication
- **Memory Reduction**: Activations memory reduced from O(L·d_model) to O(L/P·d_model)
- **Scalability**: Better performance with increasing sequence length (L > 16k tokens)
- **Overlap**: Enables computation-communication overlap between stages

## Performance Results
- **20.8% improvement** in TPS (Tokens Per Second) over baseline
- **17.6% reduction** in TPOT (Time Per Output Token) latency
- Tested on 16×H100 GPUs with 4-layer dense transformer
- Baseline: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
- Proposed: Ring Attention + Sequence Parallelism (RA+SP)

## Implementation Details
- Uses NCCL send/recv primitives or MPI point-to-point operations
- Mixed-precision (fp16/bf16) for Q, K, V tensors
- Fused kernels for projection and softmax operations
- Overlaps attention computation with asynchronous communication

## Problem Setup
- Input: X ∈ ℝ^(B×L×d_model) where B=batch size, L=sequence length, d_model=hidden size
- H attention heads, each with dimension d_h = d_model/H
- P distributed devices arranged in logical ring topology
- Objective: Compute MHA with minimal communication overhead and reduced memory footprint