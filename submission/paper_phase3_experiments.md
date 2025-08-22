# Phase 3: Experiments Extraction

## Experiments

### 1. Experimental Setup
We evaluate the proposed large-scale cross-node expert parallelism method in an **inference-only** setting using H100 GPUs. The model and configuration are as follows:

- **Model**: 4-layer Mixture-of-Experts (MoE), 16 experts per layer, each expert is a MLP
- **Precision**: FP16
- **Batch size**: 1024 tokens per forward pass
- **Dimension of MHA**: The number of heads is 16 and the dimension of each heads is 512
- **Hidden size of MLP**: The hidden is of MLP is 32768

**Metrics:**
- **TPS (Tokens per Second)**: Measures throughput
- **TPOT (Time per Output Token)**: Measures latency per token

### 2. Parallel Deployment Details

#### 2.1 Baseline Deployment (TP=8, PP=2)
- **GPUs Used**: 16 H100
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers.
  - Each pipeline stage (2 stages total) spans 8 GPUs.
  - Experts are colocated on GPUs, typically 4 experts per GPU.
- **Processing**: Tokens flow sequentially through the pipeline stages, and multiple experts per GPU share compute resources.

#### 2.2 Proposed Cross-Node Expert Parallelism
- **GPUs Used**: 64 H100 (one GPU per expert per layer)
- **Per-GPU Allocation**:
  - Each GPU hosts **exactly one expert**.
  - Tensor parallelism is applied only if a single expert's FFN cannot fit on one GPU (optional TP=2).
  - Pipeline parallelism: each MoE layer is a micro-stage; communication of tokens is overlapped with computation.
- **Routing**:
  - Input tokens are dynamically routed to the GPU holding the corresponding expert.
  - Token batches are asynchronously sent, ensuring minimal idle time.

This deployment ensures **all 64 experts per layer compute in parallel**, maximizing throughput and minimizing token latency.

### 3. Results

| Method                                 | GPUs Used | Per-GPU Deployment           | TPS (Tokens/s) | TPOT (ms) |
| -------------------------------------- | --------- | ---------------------------- | -------------- | --------- |
| Baseline (TP=8, PP=2)                  | 16        | 4 experts + TP shard per GPU | 120,000        | 8.3       |
| Proposed Cross-Node Expert Parallelism | 64        | 1 expert per GPU             | 450,000        | 2.2       |

**Notes:**
- Baseline GPUs are shared among multiple experts, causing intra-GPU contention and pipeline stalls.
- Our method dedicates one expert per GPU, enabling **maximal expert-level parallelism**.
- Throughput (TPS) is ~3.75× higher, and latency (TPOT) is ~3.8× lower than the baseline.

### 4. Discussion
- Deploying **one expert per GPU** allows full utilization of GPU compute and memory.
- Asynchronous token routing ensures minimal waiting, even across nodes.
- With 64 GPUs (unlimited H100s), the system scales near-linearly in the large EP regime (EP ≥ 16).