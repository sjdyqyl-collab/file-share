# Phase 1: Key Points Extraction

## Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Points by Section

### Introduction
- **Problem**: Traditional MoE parallelization assigns multiple experts per GPU, creating computational bottlenecks
- **Solution**: Cross-node expert parallelism with one expert per GPU
- **Key Insight**: Shift optimization focus from communication reduction to compute concurrency maximization
- **Target Environment**: HPC clusters with advanced networking (NVLink, InfiniBand, NVSwitch)

### Background
- **MoE Architecture**: Transformer variant with expert-specialized FFN layers and gating mechanism
- **Parallelism Types**: DP, TP, PP, EP - but standard EP has moderate degree with multiple experts per GPU
- **Large EP Definition**: EP ≥ 16 with one expert per GPU
- **Key Advantage**: Network bandwidth becomes less dominant than compute concurrency gains

### Methods
#### Core Components:
1. **Expert Placement Strategy**: One expert per GPU, topology-aware distribution
2. **Routing and Load Balancing**: Dynamic gating with token batching and asynchronous routing
3. **Communication Overlap**: Interleaved computation and communication with CUDA streams

#### Key Technical Details:
- **Single-Expert-Per-GPU**: Maximizes expert-level parallelism
- **Cross-Node Distribution**: Minimizes hotspotting and balances memory
- **Token Sharding**: Groups tokens by destination expert to reduce network messages
- **Pipeline Scheduling**: Immediate routing between layers with partial batch processing
- **Scalability**: Optimized for EP ≥ 16 with topology-aware routing

### Experiments
#### Setup:
- **Model**: 4-layer MoE, 16 experts/layer, MLP experts
- **Precision**: FP16
- **Batch**: 1024 sequences × 10000 tokens = 10.24M tokens/batch
- **Dimensions**: Token dim=8192, MHA=16 heads×512, MLP hidden=32768
- **GPUs**: H100 (inference-only)

#### Results:
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard/GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert/GPU | 450,000 | 2.2ms |

#### Key Findings:
- **3.75× higher throughput** (450K vs 120K TPS)
- **3.8× lower latency** (2.2ms vs 8.3ms TPOT)
- **Linear scaling** with 64 GPUs in large EP regime

### Conclusion
- **Method**: Large-scale cross-node expert parallelism with one expert per GPU
- **Benefits**: Maximized expert parallelism, balanced load, scalable communication overlap
- **Performance**: 3.75× throughput improvement, 3.8× latency reduction
- **Future Work**: Training scenarios, dynamic routing, thousands of experts