# Key Points - Large-Scale Cross-Node Expert Parallelism for MoE Models

## Abstract
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Key Innovation
- **One expert per GPU**: Maximizes expert-level parallelism by avoiding colocation
- **Large EP regime**: EP ≥ 16 for maximum independence
- **Cross-node distribution**: Exploits distributed resources fully
- **Communication-computation overlap**: Mitigates network latency

## Technical Specifications
- **Model**: 4-layer MoE, 16 experts per layer
- **Precision**: FP16
- **Batch size**: 1024 sequences
- **Sequence length**: 10000 tokens
- **Token dimension**: 8192
- **MHA**: 16 heads × 512 = 8192
- **MLP hidden size**: 32768

## Performance Results
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert per GPU | 450,000 | 2.2ms |

## Key Advantages
1. **3.75× higher throughput** (450k vs 120k TPS)
2. **3.8× lower latency** (2.2ms vs 8.3ms TPOT)
3. **Maximal expert parallelism** with no intra-GPU contention
4. **Near-linear scaling** in large EP regime
5. **Topology-aware placement** for optimal network utilization