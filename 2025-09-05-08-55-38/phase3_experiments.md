# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 (total 64 experts)
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16 (16-bit floating point)
- **Batch size**: 1024 tokens per forward pass
- **MHA configuration**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total MHA dimension: 16 × 512 = 8192
- **MLP hidden size**: 32768

### 1.2 Hardware Environment
- **GPU**: NVIDIA H100 (80GB HBM3 memory)
- **System**: High-performance computing (HPC) cluster
- **Network**: 
  - Intra-node: NVLink 4.0 (900 GB/s)
  - Inter-node: InfiniBand NDR (400 Gbps)
- **Setting**: Inference-only (no training)

### 1.3 Metrics
- **TPS (Tokens per Second)**: Primary throughput metric
- **TPOT (Time per Output Token)**: Latency metric per token

## 2. Parallel Deployment Configurations

### 2.1 Baseline Deployment (TP=8, PP=2)

#### Resource Allocation
- **Total GPUs**: 16 H100 GPUs
- **Parallelism degrees**:
  - Tensor Parallelism (TP): 8
  - Pipeline Parallelism (PP): 2
  - Expert Parallelism (EP): Not explicitly used

#### Per-GPU Configuration
- **Expert placement**: 4 experts per GPU (64 experts / 16 GPUs)
- **Tensor shard**: Each GPU holds 1/8 of tensor-parallel shard for all layers
- **Pipeline stages**: 2 stages, each spanning 8 GPUs
- **Memory usage**: ~20GB per GPU (4 experts + tensor shards)

#### Processing Flow
1. Tokens enter pipeline stage 1 (8 GPUs)
2. Each GPU processes 4 experts sequentially
3. Results transferred to pipeline stage 2 (8 GPUs)
4. Final output generated

### 2.2 Proposed Cross-Node Expert Parallelism

#### Resource Allocation
- **Total GPUs**: 64 H100 GPUs
- **Parallelism degrees**:
  - Expert Parallelism (EP): 64 (maximum possible)
  - Tensor Parallelism (TP): Optional TP=2 per expert (if memory constrained)
  - Pipeline Parallelism (PP): 4 (one micro-stage per layer)

#### Per-GPU Configuration
- **Expert placement**: Exactly 1 expert per GPU
- **Memory usage**: ~50GB per GPU (single expert + activations)
- **Network topology**: 8 nodes × 8 GPUs per node
- **Expert distribution**: 16 experts per layer × 4 layers = 64 unique experts

#### Processing Flow
1. **Token routing**: Input tokens dynamically routed to target expert GPUs
2. **Asynchronous transfer**: Token batches sent via NCCL while computation proceeds
3. **Parallel execution**: All 64 experts compute simultaneously
4. **Result collection**: Outputs gathered and routed to next layer

## 3. Experimental Results

### 3.1 Performance Comparison

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) | Efficiency |
|--------|-----------|---------------------|----------------|-----------|------------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 | 1.0× |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 | 3.75× |

### 3.2 Detailed Analysis

#### Throughput Analysis
- **Baseline**: 120,000 TPS with 16 GPUs
  - TPS per GPU: 7,500
  - Expert contention: High (4 experts sharing GPU)
- **Proposed**: 450,000 TPS with 64 GPUs
  - TPS per GPU: 7,031
  - Expert contention: None (dedicated GPU per expert)

#### Latency Analysis
- **Baseline**: 8.3ms per token
  - Pipeline stalls: Yes (sequential expert processing)
  - Communication overhead: Low (colocated experts)
- **Proposed**: 2.2ms per token
  - Pipeline stalls: Minimal (parallel expert processing)
  - Communication overhead: Amortized via overlap

#### Scalability Metrics
- **Scaling efficiency**: 93.75% (450k/480k theoretical max)
- **Network utilization**: ~75% of InfiniBand bandwidth
- **GPU utilization**: ~95% compute units active

### 3.3 Resource Utilization

#### Baseline (16 GPUs)
- **Compute utilization**: 60-70% (due to expert sharing)
- **Memory utilization**: ~25% (20GB/80GB)
- **Network utilization**: ~30% (intra-node communication)

#### Proposed (64 GPUs)
- **Compute utilization**: 95%+ (dedicated expert per GPU)
- **Memory utilization**: ~62.5% (50GB/80GB)
- **Network utilization**: 75% (cross-node communication)

## 4. Bottleneck Analysis

### 4.1 Baseline Bottlenecks
- **Intra-GPU contention**: 4 experts sharing compute resources
- **Pipeline stalls**: Sequential processing through stages
- **Memory bandwidth**: Shared between 4 experts

### 4.2 Proposed Method Bottlenecks
- **Network bandwidth**: 400 Gbps InfiniBand limit
- **Load balancing**: Uneven token distribution possible
- **Synchronization**: Cross-node coordination overhead

### 4.3 Mitigation Strategies
- **Token batching**: Reduces network messages by 16×
- **Asynchronous routing**: Overlaps communication and computation
- **Dynamic load balancing**: Adjusts gating probabilities in real-time

## 5. Validation of Large EP Regime

### 5.1 EP ≥ 16 Validation
- **Configuration**: EP=64 (well above EP=16 threshold)
- **Observation**: Linear scaling maintained up to EP=64
- **Limitation**: Network bandwidth becomes primary constraint

### 5.2 Communication vs Compute Trade-off
- **Compute-bound**: Each expert fully utilizes GPU
- **Communication-bound**: Token transfer between nodes
- **Balance**: Achieved through careful batching and overlap

## 6. Reproducibility Details

### 6.1 Environment Setup
```bash
# Hardware
8x H100 nodes, 8 GPUs per node
InfiniBand NDR 400 Gbps
NVLink 4.0 900 GB/s intra-node

# Software
CUDA 12.0+
NCCL 2.18+
PyTorch 2.0 with custom MoE kernels
FP16 precision throughout
```

### 6.2 Model Parameters
```python
model_config = {
    "num_layers": 4,
    "num_experts": 16,
    "expert_hidden_size": 32768,
    "mha_heads": 16,
    "head_dim": 512,
    "batch_size": 1024,
    "precision": "fp16"
}
```

### 6.3 Measurement Protocol
- **Warmup**: 100 iterations before measurement
- **Measurement**: 1000 iterations averaged
- **Metrics**: TPS measured at output layer, TPOT includes full pipeline latency