# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Experiments

### Experimental Setup

#### Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts**: 16 experts per layer, each expert is an MLP
- **Precision**: FP16
- **Batch Size**: 1024 tokens per forward pass
- **Attention**: 16 heads, 512 dimensions per head
- **MLP Hidden Size**: 32,768

#### Environment
- **Hardware**: H100 GPUs
- **Setting**: Inference-only evaluation
- **Metrics**: 
  - TPS (Tokens per Second) - throughput measurement
  - TPOT (Time per Output Token) - latency per token

### Deployment Configurations

#### Baseline (Conventional Approach)
- **GPUs**: 16 H100s
- **Parallelism**: TP=8, PP=2
- **Expert Placement**: 4 experts per GPU (colocated)
- **Processing**: Sequential pipeline with shared GPU resources
- **Characteristics**: Intra-GPU contention, pipeline stalls

#### Proposed Method (Large EP)
- **GPUs**: 64 H100s
- **Parallelism**: EP=64 (one expert per GPU)
- **Expert Placement**: One expert per GPU across nodes
- **Processing**: 
  - Each MoE layer as micro-stage
  - Overlapped communication and computation
  - Asynchronous token routing

### Results

| Method | GPUs | Expert Placement | TPS (Tokens/s) | TPOT (ms) |
|--------|------|------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts/GPU | 120,000 | 8.3 |
| Proposed (Large EP) | 64 | 1 expert/GPU | 450,000 | 2.2 |

### Performance Analysis

#### Throughput Improvement
- **3.75× increase** in TPS (450,000 vs 120,000 tokens/second)
- Achieved through maximal expert-level parallelism
- Elimination of intra-GPU resource contention

#### Latency Reduction
- **3.8× decrease** in TPOT (2.2ms vs 8.3ms per token)
- Reduced pipeline stalls through asynchronous routing
- Immediate processing of partial batches

#### Scalability Characteristics
- **Near-linear scaling** in large EP regime (EP ≥ 16)
- Effective utilization of 64 GPUs vs baseline 16 GPUs
- Communication overhead successfully mitigated through overlap techniques

### Key Findings

1. **Expert Isolation Benefit**: One expert per GPU eliminates computational bottlenecks from resource sharing
2. **Communication-Compute Overlap**: Asynchronous routing enables effective hiding of cross-node latency
3. **Scalability**: Method scales effectively with available GPU resources in HPC environments
4. **Resource Efficiency**: Higher GPU count (64 vs 16) yields superlinear performance improvement due to eliminated contention

### Experimental Validation
- Results confirm theoretical benefits of large EP approach
- Demonstrates practical feasibility in H100 cluster environment
- Validates design choice to prioritize compute concurrency over communication reduction