# Helix: Two-Level Attention Partitioning - Phase 3: Experiments

## Experimental Setup

### Hardware Configuration
- **System**: 16 NVIDIA H100 GPUs
- **Precision**: Mixed precision (FP16)
- **Batch Size**: 1024 (fixed)
- **Attention Configuration**: 16 heads, 512 dimension per head
- **MLP Hidden Size**: 32768

### Model Architectures Tested
1. **4-layer Dense Transformer**
2. **4-layer Mixture-of-Experts (MoE) Transformer** - 8 experts per layer

### Baseline Configuration
- **Tensor Parallelism (TP)**: Degree 8
- **Pipeline Parallelism (PP)**: Degree 2
- **Total GPUs utilized**: TP=8 + PP=2 = 16 GPUs
- **Baseline designation**: Standard large-scale deployment method

### Proposed Method Configuration
- **Partitioning**: m×n = 16 (m=4, n=4 configuration)
- **Total partitions**: 16 (matching 16 GPUs)
- **Mapping**: One partition per GPU

## Evaluation Metrics
- **Throughput (TPS)**: Tokens processed per second
- **Time Per Output Token (TPOT)**: Average synchronization and communication overhead per token (milliseconds)

## Experimental Results

### Dense Transformer Results
| Method | TPS (tokens/sec) | TPOT (ms) | Improvement |
|--------|------------------|-----------|-------------|
| Baseline (TP=8, PP=2) | 1,200,000 | 0.35 | - |
| Proposed (m×n=16) | 1,580,000 | 0.22 | +31.7% TPS, -37.1% TPOT |

### MoE Transformer Results
| Method | TPS (tokens/sec) | TPOT (ms) | Improvement |
|--------|------------------|-----------|-------------|
| Baseline (TP=8, PP=2) | 850,000 | 0.45 | - |
| Proposed (m×n=16) | 1,150,000 | 0.30 | +35.3% TPS, -33.3% TPOT |

## Performance Analysis

### Throughput Improvements
- **Dense model**: 31.7% increase (1.2M → 1.58M tokens/sec)
- **MoE model**: 35.3% increase (850K → 1.15M tokens/sec)
- **Average improvement**: 33.5% across both model types

### Communication Overhead Reduction
- **Dense model**: 37.1% reduction in TPOT (0.35ms → 0.22ms)
- **MoE model**: 33.3% reduction in TPOT (0.45ms → 0.30ms)
- **Average reduction**: 35.2% across both model types

### Hardware Utilization
- **Full GPU utilization**: 16/16 GPUs actively processing
- **Load balancing**: Even distribution via m×n partitioning
- **Memory distribution**: Each GPU handles 1/16 of total computation

## DAG Generation Requirements

### Input Data Flow
1. **Input tensor**: X ∈ ℝ^(1024×L×8192) [batch=1024, hidden=16×512]
2. **Weight distribution**: W_Q, W_K, W_V each split into 16 partitions
3. **Device assignment**: Each GPU gets 1/16 of weight matrices

### Computation DAG
```
Level 1: Input Distribution
├── GPU 0: X → Q^(0,0), K^(0,0), V^(0,0)
├── GPU 1: X → Q^(0,1), K^(0,1), V^(0,1)
├── ...
└── GPU 15: X → Q^(3,3), K^(3,3), V^(3,3)

Level 2: Parallel Attention Computation
├── GPU 0: Attention^(0,0) = softmax(Q^(0,0)(K^(0,0))^T/√128)V^(0,0)
├── GPU 1: Attention^(0,1) = softmax(Q^(0,1)(K^(0,1))^T/√128)V^(0,1)
├── ...
└── GPU 15: Attention^(3,3) = softmax(Q^(3,3)(K^(3,3))^T/√128)V^(3,3)

Level 3: Hierarchical Aggregation
├── Intra-group (4 GPUs each):
│   ├── Group 0: Concat(Attention^(0,0-3))
│   ├── Group 1: Concat(Attention^(1,0-3))
│   ├── Group 2: Concat(Attention^(2,0-3))
│   └── Group 3: Concat(Attention^(3,0-3))
└── Inter-group: Concat(Group 0-3 outputs)
```

### Communication Pattern for DAG
- **All-gather operations**: Within each head group (4 GPUs)
- **Concatenation**: Across 4 head groups
- **Synchronization points**: After intra-group and inter-group concatenation
- **Data transfer**: 1/16 → 1/4 → full tensor reconstruction

## Reproducibility Parameters
- **Random seed**: Fixed for consistent benchmarking
- **Warmup iterations**: 100 steps before measurement
- **Measurement duration**: 1000 steps averaged
- **Environment**: CUDA 12.0, NCCL 2.18, PyTorch 2.0
- **Network**: NVLink + InfiniBand interconnect

## Key Findings for Deployment
1. **Optimal partitioning**: m×n=16 provides best utilization for 16 GPUs
2. **Load balancing**: Equal work per GPU (both dense and MoE)
3. **Communication efficiency**: Hierarchical aggregation reduces sync time
4. **Memory efficiency**: 16× reduction in per-GPU memory usage
5. **Scalability**: Method scales beyond head count limitations