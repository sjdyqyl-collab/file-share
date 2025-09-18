# Phase 1: Key Points Extraction

## Key Points from the Paper

### 1. Core Problem
Traditional MoE parallelization strategies assign multiple experts to the same GPU, creating computational bottlenecks and limiting expert-level parallelism as model and cluster sizes grow.

### 2. Proposed Solution
Large-scale cross-node expert parallelism strategy that:
- Deploys at most one expert per GPU
- Distributes experts across nodes to exploit available compute resources
- Focuses on Expert Parallelism (EP) ≥ 16 ("large EP")

### 3. Key Innovations
- **Single-Expert-Per-GPU Deployment**: Ensures minimal contention and high compute efficiency
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory
- **Asynchronous Token Routing**: Overlaps communication with computation
- **Load Balancing**: Dynamic gating probabilities to prevent expert overloading

### 4. Technical Components
1. **Expert Placement Strategy**: Assigning experts across GPUs and nodes
2. **Routing and Load Balancing**: Ensuring balanced input distribution to experts
3. **Communication Overlap and Scheduling**: Minimizing cross-node data transfer impact

### 5. Experimental Results
- **Model**: 4-layer MoE, 16 experts per layer (64 total experts)
- **Configuration**: 64 H100 GPUs, 1 expert per GPU
- **Performance**: 450,000 TPS vs 120,000 TPS (baseline)
- **Improvement**: 3.75× higher throughput, 3.8× lower latency
- **Baseline**: TP=8, PP=2 with 16 GPUs, 4 experts per GPU

### 6. Scalability Benefits
- Near-linear scaling in large EP regime (EP ≥ 16)
- Full GPU utilization for compute
- Effective communication cost amortization across many tokens
- Compatible with tensor parallelism and data parallelism for very large models