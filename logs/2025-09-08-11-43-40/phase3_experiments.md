# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (16-bit floating point)

### 1.2 Input Configuration
- **Batch size**: 1024 sequences per batch
- **Sequence length**: 10,000 tokens per sequence
- **Total tokens per batch**: 10,240,000 tokens (1024 × 10,000)
- **Token dimension**: 8192 dimensions per token

### 1.3 Attention Configuration
- **Multi-Head Attention (MHA)**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total attention dimension: 16 × 512 = 8192 (matches token dimension)

### 1.4 MLP Configuration
- **Hidden size of MLP**: 32,768
- **Expert architecture**: Standard transformer FFN with expansion factor 4 (8192 → 32768 → 8192)

### 1.5 Hardware Environment
- **GPU type**: NVIDIA H100
- **Environment**: High-performance computing (HPC) cluster
- **Network**: Modern interconnects (NVLink, InfiniBand, H100-class NVSwitch)

## 2. Evaluation Metrics

### 2.1 Throughput Metrics
- **TPS (Tokens per Second)**: Primary throughput measure
  - Baseline target: 120,000 TPS
  - Proposed target: 450,000 TPS

### 2.2 Latency Metrics
- **TPOT (Time per Output Token)**: Average latency per token
  - Baseline: 8.3 ms
  - Proposed: 2.2 ms

## 3. Parallel Deployment Configurations

### 3.1 Baseline Configuration
- **GPUs used**: 16 H100 GPUs
- **Parallelism strategy**:
  - Tensor Parallelism (TP): 8
  - Pipeline Parallelism (PP): 2
  - Expert Parallelism (EP): 2 (16 experts / 8 GPUs per stage)
- **Per-GPU allocation**:
  - Each GPU holds 1/8 of tensor-parallel shard for all layers
  - Pipeline stages: 2 stages total, each spanning 8 GPUs
  - Expert placement: 4 experts per GPU (16 experts / 4 GPUs per stage)
- **Processing flow**: Tokens flow sequentially through pipeline stages with shared compute resources

### 3.2 Proposed Cross-Node Expert Parallelism
- **GPUs used**: 64 H100 GPUs
- **Parallelism strategy**:
  - Expert Parallelism (EP): 64 (16 experts × 4 layers = 64 total expert instances)
  - Tensor Parallelism (TP): 1 (no TP within expert, optional TP=2 if needed)
  - Pipeline Parallelism (PP): 4 (each MoE layer as micro-stage)
- **Per-GPU allocation**:
  - Each GPU hosts exactly one expert instance
  - Total experts: 64 (16 experts/layer × 4 layers)
  - Expert replication: Each expert appears once across the 64-GPU deployment
- **Routing mechanism**:
  - Dynamic token routing to GPU holding corresponding expert
  - Asynchronous token batch transmission
  - Overlap computation with communication

## 4. Experimental Results

### 4.1 Performance Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) | Improvement |
|--------|-----------|-------------------|----------------|-----------|-------------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 | - |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 | 3.75× TPS, 3.8× TPOT |

### 4.2 Analysis of Results
- **Throughput improvement**: 3.75× increase (120k → 450k TPS)
- **Latency reduction**: 3.8× decrease (8.3ms → 2.2ms TPOT)
- **Resource utilization**: Baseline suffers from intra-GPU contention and pipeline stalls
- **Scalability**: Proposed method achieves near-linear scaling with 64 GPUs

### 4.3 Bottleneck Analysis
- **Baseline bottlenecks**:
  - Multiple experts sharing GPU compute resources
  - Pipeline stalls between stages
  - Limited expert-level parallelism (EP=2)
- **Proposed advantages**:
  - Full GPU compute utilization per expert
  - Maximal expert-level parallelism (EP=64)
  - Asynchronous communication hiding latency

## 5. Discussion Points

### 5.1 Scalability Validation
- Large EP regime (EP ≥ 16) successfully demonstrated with EP=64
- Near-linear scaling achieved with abundant GPU resources
- Communication overhead effectively mitigated through overlapping

### 5.2 Resource Requirements
- **GPU scaling**: 4× increase in GPUs (16 → 64) yields 3.75× throughput gain
- **Network efficiency**: Modern interconnects sufficient to sustain high bandwidth
- **Memory efficiency**: Balanced expert placement prevents memory hotspots

### 5.3 Practical Implications
- **HPC environments**: Particularly effective in large GPU clusters
- **Cost considerations**: Requires 4× more GPUs but achieves 3.75× performance
- **Future applicability**: Blueprint for scaling to even larger configurations