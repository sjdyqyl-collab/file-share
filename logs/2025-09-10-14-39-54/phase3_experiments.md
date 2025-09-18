# Phase Three: Experiments

## 1. Experimental Setup
### Model and Input
- **Model**: 4-layer MoE, 16 experts/layer (total 64 experts), FP16 precision.
- **MLP Details**: Hidden size = 32768, token dimension = 8192.
- **MHA Details**: 16 heads, 512 dim/head (total MHA dim = 8192).
- **Input**: 1024 sequences/batch, 10000 tokens/sequence.

### Hardware
- GPUs: H100 (80GB VRAM).
- **Baseline**: 16 GPUs.
- **Proposed**: 64 GPUs.

### Metrics
- **TPS (Tokens per Second)**: Throughput (higher = better).
- **TPOT (Time per Output Token)**: Latency (lower = better).

## 2. Parallel Deployment Details
### 2.1 Baseline Deployment (TP=8, PP=2)
- **GPUs**: 16 (2 pipeline stages × 8 GPUs/stage).
- **Per-GPU Allocation**: 1/8 tensor-parallel shard (all layers) + 4 experts/GPU.
- **Processing**: Sequential pipeline stages; experts share GPU resources.

### 2.2 Proposed Deployment (EP=16, 1 Expert/GPU)
- **GPUs**: 64 (1 GPU/expert/layer × 4 layers × 16 experts/layer).
- **Per-GPU Allocation**: 1 expert (no TP unless expert > GPU memory) + token routing logic.
- **Processing**: Asynchronous token routing to experts; compute/communication overlap.

## 3. Results
| Method                                 | GPUs Used | Per-GPU Deployment           | TPS (Tokens/s) | TPOT (ms) |
| -------------------------------------- | --------- | ---------------------------- | -------------- | --------- |
| Baseline (TP=8, PP=2)                  | 16        | 4 experts + 1/8 TP shard     | 120,000        | 8.3       |
| Proposed Cross-Node Expert Parallelism | 64        | 1 expert                     | 450,000        | 2.2       |

## 4. Discussion
- **Baseline Limitations**: Intra-GPU expert contention and pipeline stalls reduce throughput.
- **Proposed Gains**: 3.75× TPS and 3.8× TPOT improvements from full expert parallelism and communication overlap.
- **Scalability**: Near-linear scaling with 64 GPUs (EP=16), validating large EP effectiveness.

## Key Takeaways
- One-expert-per-GPU eliminates contention and maximizes compute utilization.
- Asynchronous routing and communication overlap mitigate network bottlenecks.
- Large EP (≥16) is critical for scaling MoE models in HPC environments.