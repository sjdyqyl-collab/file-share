# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## 1. Introduction
Traditional MoE parallelization assigns multiple experts to the same GPU to reduce inter-node communication, creating computational bottlenecks as cluster sizes grow. We present a cross-node expert parallelism method that prioritizes distributing experts across nodes such that each GPU hosts at most one expert, pushing Expert Parallelism (EP) to 16 or beyond to unlock higher degrees of concurrent computation.

## 2. Methods

### 2.1 Expert Placement Strategy
- **Single-Expert-Per-GPU**: Each GPU hosts at most one expert
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, and memory
- **Constraint**: E experts across G GPUs, ensuring distinct GPU assignment when E ≤ G

### 2.2 Routing and Load Balancing
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with computation
- **Dynamic Load Balancing**: Monitor per-expert load and adjust gating probabilities

### 2.3 Communication Overlap and Scheduling
- **Compute-Communication Interleaving**: Process one batch while transferring the next
- **Pipeline Scheduling**: Each MoE layer as a micro-stage with immediate routing
- **Implementation**: CUDA streams or NCCL/MPI for non-blocking communication

### 2.4 Scalability Considerations
- **Large EP Regime**: EP ≥ 16 with network bandwidth as primary limiting factor
- **Integration**: Compatible with tensor parallelism (TP) and data parallelism (DP)
- **Memory Handling**: Optional TP=2 within single expert if memory constrained

## 3. Experiments

### 3.1 Setup
- **Model**: 4-layer MoE, 16 experts/layer, MLP experts
- **Precision**: FP16
- **Batch**: 1024 sequences × 10000 tokens
- **Dimensions**: Token=8192, MLP hidden=32768, MHA=16×512

### 3.2 Deployments

**Baseline (TP=8, PP=2)**:
- GPUs: 16 H100
- Configuration: 4 experts per GPU, 2 pipeline stages
- Experts colocated with shared compute resources

**Proposed Cross-Node Expert Parallelism**:
- GPUs: 64 H100
- Configuration: 1 expert per GPU, EP=64
- Asynchronous token routing with overlapped communication

### 3.3 Results
| Method | GPUs | TPS | TPOT (ms) |
|--------|------|-----|-----------|
| Baseline | 16 | 120,000 | 8.3 |
| Proposed | 64 | 450,000 | 2.2 |

**Performance**: 3.75× higher throughput, 3.8× lower latency

## 4. Conclusion
Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU. Results demonstrate 3.75× higher throughput and 3.8× lower latency compared to traditional approaches, providing a scalable blueprint for high-performance MoE inference in GPU-rich environments.