# Keypoints Extraction - Large-Scale Cross-Node Expert Parallelism for MoE Models

## Core Contribution
A large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models that maximizes computational parallelism by deploying at most one expert per GPU.

## Key Innovations

### 1. Single-Expert-Per-GPU Deployment
- Contrary to conventional approaches that colocate multiple experts on the same device
- Ensures minimal expert-level contention and improved throughput
- Each expert processes tokens without contention from other experts on the same device

### 2. Large Expert Parallelism (Large EP) Definition
- Defines "large EP" as configurations where EP ≥ 16
- Significantly increases independence of expert computation
- Enables better scalability and reduced inter-expert interference

### 3. Cross-Node Distribution Strategy
- Topology-aware placement considering node-to-node bandwidth and latency
- GPU memory capacity per node consideration
- Expected token routing patterns analysis
- Minimizes hotspotting on any single node

### 4. Asynchronous Communication Overlap
- Interleaves expert computation and communication
- Uses CUDA streams or asynchronous communication libraries (NCCL/MPI)
- Pipeline scheduling for multi-layer MoE networks
- Tokens processed while next batch is transferred

### 5. Dynamic Load Balancing
- Token batching by destination expert to reduce network messages
- Asynchronous routing to overlap with expert computation
- Dynamic gating probability adjustment to prevent expert overloading

## Performance Results
- **Throughput**: 3.75× higher (450,000 vs 120,000 tokens/s)
- **Latency**: 3.8× lower (2.2ms vs 8.3ms TPOT)
- **Configuration**: 16 H100 GPUs, 4-layer MoE, 16 experts per layer

## Technical Specifications
- Model: 4-layer MoE, 16 experts per layer
- Precision: FP16
- Batch size: 1024 sequences
- Sequence length: 10000 tokens
- Token dimension: 8192
- MHA: 16 heads, 512 dimension per head
- MLP hidden size: 32768

## Deployment Comparison
- **Baseline**: TP=8, PP=2, 16 GPUs, 8 experts per GPU + TP shard
- **Proposed**: 16 GPUs, 1 expert per GPU, maximal expert-level parallelism

## Scalability Features
- Compatible with tensor model parallelism (TP) for large experts
- Integrates with data parallelism (DP) for synchronized weight updates
- Optimized for HPC and large GPU cluster environments
- Near-linear scaling in large EP regime (EP ≥ 16)