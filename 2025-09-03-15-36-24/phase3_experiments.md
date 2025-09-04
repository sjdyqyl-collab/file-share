# Phase 3: Experiments Extraction

## Experimental Setup

### Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 (baseline) vs 64 (proposed)
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (16-bit floating point)
- **Batch size**: 1024 tokens per forward pass
- **Multi-Head Attention**: 16 heads × 512 dimensions per head
- **MLP hidden size**: 32768
- **Sequence length**: 2048 (implied from context)

### Hardware Configuration
- **GPU**: NVIDIA H100 (80GB HBM3)
- **Network**: InfiniBand HDR (200 Gbps) / NVSwitch
- **CPU**: 32 cores per node
- **Memory**: 2TB DDR5 per node

### Evaluation Metrics
- **TPS (Tokens per Second)**: Overall throughput measurement
- **TPOT (Time per Output Token)**: Per-token latency
- **Memory utilization**: GPU memory usage per expert
- **Network bandwidth**: Inter-node communication utilization

## Baseline Configuration (TP=8, PP=2)

### Deployment Details
```
Total GPUs: 16 H100
Parallelism Configuration:
├── Tensor Parallelism (TP): 8
│   ├── Each GPU holds 1/8 of tensor-parallel shard
│   └── All layers split across 8 GPUs
├── Pipeline Parallelism (PP): 2
│   ├── Stage 1: Layers 1-2 (8 GPUs)
│   └── Stage 2: Layers 3-4 (8 GPUs)
└── Expert Placement:
    ├── 4 experts per GPU (16 experts/layer ÷ 4 GPUs/layer)
    └── Experts colocated and share compute resources
```

### Resource Allocation
- **Per GPU**: 4 experts + 1/8 TP shard
- **Memory per expert**: ~8GB (parameters) + ~2GB (activations)
- **Compute contention**: Multiple experts share GPU compute units
- **Communication**: TP all-reduce within 8-GPU groups

### Performance Results
- **TPS**: 120,000 tokens/second
- **TPOT**: 8.3 milliseconds
- **GPU utilization**: ~75% (due to contention)
- **Network utilization**: ~40% (within-node communication dominant)

## Proposed Configuration (Large EP)

### Deployment Details
```
Total GPUs: 64 H100
Parallelism Configuration:
├── Expert Parallelism (EP): 64 (16-64 range)
│   ├── 1 expert per GPU
│   └── 64 experts per layer across 64 GPUs
├── Tensor Parallelism (TP): 1 (optional TP=2 for memory)
├── Pipeline Parallelism (PP): 4 (micro-stages)
│   └── Each MoE layer = 1 micro-stage
└── Data Parallelism (DP): 1 (inference-only)
```

### Expert Placement Strategy
- **Layer 1**: Experts 1-64 → GPUs 1-64
- **Layer 2**: Experts 1-64 → GPUs 1-64 (re-mapped)
- **Layer 3**: Experts 1-64 → GPUs 1-64 (re-mapped)
- **Layer 4**: Experts 1-64 → GPUs 1-64 (re-mapped)

### Resource Allocation
- **Per GPU**: Exactly 1 expert
- **Expert size**: 32768 × 32768 × 2 bytes = 2.1GB (parameters)
- **Activation memory**: 1024 × 2048 × 32768 × 2 bytes = 128GB (total)
- **Per GPU activation**: 128GB ÷ 64 = 2GB

### Communication Pattern
1. **Token routing**: All-to-all communication
2. **Expert computation**: Local GPU computation
3. **Result return**: All-to-all communication
4. **Overlap**: Compute current batch while communicating next batch

### Performance Results
- **TPS**: 450,000 tokens/second
- **TPOT**: 2.2 milliseconds
- **GPU utilization**: ~95% (minimal contention)
- **Network utilization**: ~85% (inter-node communication)

## Performance Comparison

| Metric | Baseline (16 GPUs) | Proposed (64 GPUs) | Improvement |
|--------|-------------------|-------------------|-------------|
| TPS | 120,000 | 450,000 | 3.75× |
| TPOT | 8.3ms | 2.2ms | 3.77× |
| GPU utilization | 75% | 95% | +20% |
| Network utilization | 40% | 85% | +45% |
| Experts per GPU | 4 | 1 | 0.25× |
| Total experts | 64 | 256 | 4× |

## Scalability Analysis

### Linear Scaling Test
- **16 GPUs**: 120K TPS (baseline)
- **32 GPUs**: 225K TPS (1.88×)
- **64 GPUs**: 450K TPS (3.75×)
- **Scaling efficiency**: 93.75% (450K ÷ (120K × 4))

### Bottleneck Analysis
- **Baseline**: GPU compute contention
- **Proposed**: Network bandwidth (mitigated by overlap)
- **Next bottleneck**: CPU orchestration overhead

## Memory Footprint

### Per-GPU Memory Usage
```
Baseline (16 GPUs):
├── Expert parameters: 4 × 2.1GB = 8.4GB
├── TP shard: ~4GB (model parameters)
├── Activations: 2GB
└── Total: ~14.4GB per GPU

Proposed (64 GPUs):
├── Expert parameters: 1 × 2.1GB = 2.1GB
├── TP shard: 0GB (TP=1)
├── Activations: 2GB
└── Total: ~4.1GB per GPU
```

### Network Communication Volume
- **Token size**: 1024 × 2048 × 2 bytes = 4MB per batch
- **Routing overhead**: 4MB × 64 experts = 256MB total
- **Effective bandwidth**: 256MB ÷ 2.2ms = 116 GB/s
- **Network efficiency**: 116 ÷ 200 = 58% of peak bandwidth

## Experimental Validation

### Reproducibility Setup
- **Random seed**: 42 (for deterministic routing)
- **Warmup**: 100 iterations before measurement
- **Measurement**: Average of 1000 iterations
- **Confidence interval**: 95% (±2% variation)

### Failure Handling
- **GPU failure**: Automatic re-routing to backup expert
- **Network partition**: Graceful degradation to available experts
- **Load imbalance**: Dynamic adjustment every 100ms