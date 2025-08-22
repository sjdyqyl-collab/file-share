# Helix: Two-Level Attention Partitioning - Experiments

## Abstract (Retained in full)
We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.

## Experimental Setup

### Hardware Configuration
- **System**: 16 NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16)
- **Framework**: Compatible with existing model parallel frameworks

### Model Configurations
1. **4-layer Dense Transformer model**
2. **4-layer Mixture-of-Experts (MoE) Transformer model** - 8 experts per layer

### Fixed Parameters
- **Batch size**: 1024
- **Number of heads**: 16
- **Head dimension**: 512
- **MLP hidden size**: 32768

### Baseline Configuration
- **Tensor Parallelism (TP)**: degree 8
- **Pipeline Parallelism (PP)**: degree 2
- **Total devices**: 16 (TP=8 + PP=2)

### Proposed Configuration
- **Two-level partitioning**: m×n = 16 partitions
- **Mapping**: 16 partitions → 16 devices

## Metrics
- **Throughput (TPS)**: Tokens processed per second
- **Time Per Output Token (TPOT)**: Average synchronization and communication overhead per token (milliseconds)

## Results

| Model Type    | Method                | TPS (tokens/sec) | TPOT (ms) |
| ------------- | --------------------- | ---------------- | --------------- |
| 4-layer Dense | Baseline (TP=8, PP=2) | 1,200,000        | 0.35            |
| 4-layer Dense | Proposed (m×n=16)     | 1,580,000        | 0.22            |
| 4-layer MoE   | Baseline (TP=8, PP=2) | 850,000          | 0.45            |
| 4-layer MoE   | Proposed (m×n=16)     | 1,150,000        | 0.30            |

## Analysis

### Performance Improvements
- **Dense model**: 
  - Throughput: +31.7% (1.2M → 1.58M tokens/sec)
  - Communication overhead: -37.1% (0.35ms → 0.22ms TPOT)
- **MoE model**:
  - Throughput: +35.3% (850K → 1.15M tokens/sec)
  - Communication overhead: -33.3% (0.45ms → 0.30ms TPOT)

### Key Findings
- Two-level partitioning fully exploits 16 GPUs by mapping m×n=16 partitions to devices
- Finer granularity enables better load balancing vs traditional TP+PP
- Reduced synchronization cost through efficient communication patterns
- Large batch size (1024) and FP16 precision ensure GPU saturation
- Performance gains attributed to parallelization strategy, not hardware idling

## DAG Generation Requirements

### Experimental DAG Components
1. **Input Preparation**: Batch size 1024, sequence processing
2. **Model Loading**: 4-layer Dense or MoE Transformer
3. **Partitioning Setup**: Configure m×n=16 partitions
4. **Device Mapping**: Assign 16 partitions to 16 H100 GPUs
5. **Execution Monitoring**: Measure TPS and TPOT metrics
6. **Result Collection**: Aggregate performance statistics

### Deployment Verification
- Verify 16-way partitioning maps correctly to 16 devices
- Confirm FP16 precision maintains numerical stability
- Validate communication overhead measurements
- Ensure proper load balancing across all devices