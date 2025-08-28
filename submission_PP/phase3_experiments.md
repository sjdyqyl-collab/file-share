# Phase 3: Experiments Extraction

## Setup
We evaluate our proposed layer-wise deployment method for large models in the inference stage. The hardware platform consists of 16 NVIDIA H100 GPUs. We use two model types:
- **Dense model:** A 16-layer fully connected dense network
- **MoE model:** A 16-layer mixture-of-experts (MoE) model with 8 experts per layer

Both models use FP16 precision and are tested with a batch size of 1024. The number of head is fixed at 16, the dimension of each head is fixed at 512, the hidden size of MLP is fixed at 32768. The baseline comparison is a standard tensor parallelism (TP) and pipeline parallelism (PP) setup, specifically TP=8 and PP=2, which fully utilizes the 16 GPUs (8 × 2 = 16).

We measure performance with two key metrics:
- **Tokens Per Second (TPS):** The number of output tokens generated per second
- **Time Per Output Token (TPOT):** The average time to produce a single output token, in milliseconds

## Results

| Model                    | Method                | GPUs | TPS (tokens/s) | TPOT (ms) |
|-------------------------|-----------------------|------|----------------|-----------|
| Dense (16-layer)        | Baseline (TP=8, PP=2) | 16   | 12,800         | 0.078     |
| Dense (16-layer)        | Proposed Layer-wise   | 16   | 15,360         | 0.065     |
| MoE (16-layer, 8 experts) | Baseline (TP=8, PP=2) | 16   | 10,200         | 0.098     |
| MoE (16-layer, 8 experts) | Proposed Layer-wise   | 16   | 13,400         | 0.075     |

## Analysis
- For the dense model, our proposed deployment method achieves a **20% increase in TPS** and a corresponding **17% reduction in TPOT** compared to the baseline. This improvement results from more efficient on-chip memory utilization, reducing memory access latency.

- For the MoE model, which is more complex and has irregular computation patterns due to expert routing, our method achieves an even larger gain, with **approximately 31% higher TPS** and **23% faster TPOT**. The improved layer partitioning and cache fitting helps mitigate communication overheads that typically limit MoE model scaling.

- The baseline TP=8, PP=2 approach is effective but does not consider on-chip memory constraints explicitly, leading to more off-chip memory accesses and communication delays.