# Phase 3: Experiments Extraction

## 1. Experimental Setup

### Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16
- **Batch size**: 1024 tokens per forward pass
- **Multi-head attention**: 16 heads, 512 dimension per head
- **MLP hidden size**: 32,768

### Hardware Environment
- **GPUs**: H100 (NVIDIA Hopper architecture)
- **Setting**: Inference-only (no training)

### Evaluation Metrics
- **TPS (Tokens per Second)**: Measures overall throughput
- **TPOT (Time per Output Token)**: Measures latency per token

## 2. Parallel Deployment Configurations

### 2.1 Baseline Deployment (TP=8, PP=2)
- **Total GPUs**: 16 H100
- **Parallelism configuration**:
  - **Tensor Parallelism (TP)**: 8 (each GPU holds 1/8 of tensor-parallel shard)
  - **Pipeline Parallelism (PP)**: 2 stages
  - **Expert Parallelism (EP)**: Not explicitly stated (experts colocated)
- **Per-GPU allocation**:
  - Each GPU holds 1/8 of tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs
  - **Experts per GPU**: 4 experts colocated on each GPU
- **Processing flow**: Tokens flow sequentially through pipeline stages
- **Limitation**: Multiple experts per GPU share compute resources, causing contention

### 2.2 Proposed Cross-Node Expert Parallelism
- **Total GPUs**: 64 H100
- **Parallelism configuration**:
  - **Expert Parallelism (EP)**: 64 (one expert per GPU)
  - **Tensor Parallelism (TP)**: Optional TP=2 if single expert cannot fit on one GPU
  - **Pipeline Parallelism (PP)**: Each MoE layer as micro-stage
- **Per-GPU allocation**:
  - **One expert per GPU**: Each GPU hosts exactly one expert
  - **Total experts per layer**: 64 (16 experts × 4 layers)
  - **Expert memory**: Full expert fits on single GPU
- **Routing mechanism**:
  - **Dynamic routing**: Input tokens routed to GPU holding corresponding expert
  - **Asynchronous communication**: Token batches sent asynchronously to minimize idle time
  - **Overlap**: Communication overlapped with computation

## 3. Experimental Results

### Performance Comparison Table
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### Performance Improvements
- **Throughput improvement**: 450,000 ÷ 120,000 = **3.75× higher TPS**
- **Latency reduction**: 8.3 ÷ 2.2 = **3.8× lower TPOT**
- **GPU utilization**: All 64 GPUs fully utilized with one expert per GPU
- **Scalability**: Near-linear scaling achieved in large EP regime (EP=64 ≥ 16)

## 4. Analysis and Discussion

### Baseline Limitations
- **Intra-GPU contention**: 4 experts per GPU share compute resources
- **Pipeline stalls**: Sequential processing through pipeline stages
- **Resource underutilization**: GPU compute not fully utilized due to expert sharing

### Proposed Method Advantages
- **Maximal expert parallelism**: All 64 experts compute simultaneously
- **No expert contention**: One expert per GPU eliminates resource sharing
- **Asynchronous overlap**: Communication and computation fully overlapped
- **Linear scaling**: Performance scales with available GPUs in large EP regime

### Network Considerations
- **Communication overhead**: Mitigated by asynchronous token routing
- **Bandwidth utilization**: Modern HPC networking (NVLink, InfiniBand, NVSwitch) supports high bandwidth
- **Topology awareness**: Expert placement considers node-to-node bandwidth and latency

## 5. Experimental Validation
- **Setting**: Inference-only evaluation
- **Model size**: 4-layer MoE with 64 total experts
- **Precision**: FP16 for memory efficiency
- **Batch processing**: 1024 tokens processed per forward pass
- **Hardware**: H100 GPU cluster with sufficient scale (64 GPUs)

## 6. Key Experimental Insights
1. **Expert isolation**: One expert per GPU eliminates contention and maximizes compute efficiency
2. **Communication-computation overlap**: Asynchronous routing enables near-linear scaling
3. **Large EP effectiveness**: EP=64 demonstrates practical benefits of large-scale expert parallelism
4. **Scalability proof**: Results validate theoretical benefits of large EP regime (EP ≥ 16)