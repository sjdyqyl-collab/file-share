# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration

**Architecture Details**:
- **Model Type**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Architecture**: Multi-Layer Perceptron (MLP)
- **Precision**: FP16 (16-bit floating point)
- **Total Parameters**: 64 experts × (8192 × 32768 + 32768 × 8192) = 33.5B parameters

### 1.2 Input Configuration

**Batch Specifications**:
- **Batch Size**: 1024 sequences per batch
- **Sequence Length**: 10,000 tokens per sequence
- **Token Dimension**: 8,192 dimensional embeddings
- **Total Tokens per Batch**: 1024 × 10,000 = 10.24M tokens

### 1.3 Multi-Head Attention Details

**Attention Configuration**:
- **Number of Heads**: 16 attention heads
- **Head Dimension**: 512 per head
- **Total MHA Dimension**: 16 × 512 = 8,192 (matches token dimension)

### 1.4 MLP Expert Details

**Expert Architecture**:
- **Input Dimension**: 8,192
- **Hidden Dimension**: 32,768
- **Output Dimension**: 8,192
- **Activation Function**: GELU
- **Expert Parameters**: (8192 × 32768) + (32768 × 8192) = 536.9M per expert

### 1.5 Hardware Configuration

**GPU Specifications**:
- **GPU Type**: NVIDIA H100
- **GPU Memory**: 80GB HBM3 per GPU
- **Interconnect**: NVLink 4.0, InfiniBand HDR
- **Network Bandwidth**: 450 GB/s aggregate

### 1.6 Evaluation Metrics

**Performance Metrics**:
- **TPS (Tokens per Second)**: Total throughput measurement
- **TPOT (Time per Output Token)**: Average latency per token
- **Calculation**: TPOT = 1 / (TPS / total_tokens_in_batch)

## 2. Parallel Deployment Details

### 2.1 Baseline Deployment (TP=8, PP=2)

**Configuration**:
- **GPUs Used**: 16 H100 GPUs
- **Tensor Parallelism (TP)**: 8-way split
- **Pipeline Parallelism (PP)**: 2 stages
- **Expert Parallelism (EP)**: Not explicitly used

**Per-GPU Allocation**:
- **Tensor Sharding**: Each GPU holds 1/8 of tensor-parallel shard for all layers
- **Pipeline Stages**: 2 stages, each spanning 8 GPUs
- **Expert Placement**: 4 experts per GPU (colocated)
- **Memory Usage**: ~50GB per GPU (shared among 4 experts)

**Processing Flow**:
1. Tokens flow sequentially through pipeline stages
2. Within each stage, tensor parallelism splits computation
3. Multiple experts share GPU compute resources
4. Expert contention occurs due to colocation

### 2.2 Proposed Cross-Node Expert Parallelism

**Configuration**:
- **GPUs Used**: 64 H100 GPUs
- **Expert Parallelism (EP)**: 64-way (EP=64)
- **Tensor Parallelism (TP)**: Optional TP=2 within expert if needed
- **Pipeline Parallelism (PP)**: Micro-stage per layer

**Per-GPU Allocation**:
- **Expert Placement**: Exactly one expert per GPU
- **Memory Usage**: ~537MB per expert + 2.5GB token buffer
- **No Expert Sharing**: Complete isolation between experts
- **Tensor Parallelism**: Applied only if expert exceeds GPU memory

**Processing Flow**:
1. **Input Distribution**: Tokens routed to 64 experts in parallel
2. **Parallel Computation**: All 64 experts compute simultaneously
3. **Asynchronous Communication**: Token transfers overlap with computation
4. **Result Collection**: Gather results from all experts

## 3. Experimental Results

### 3.1 Performance Comparison

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### 3.2 Performance Analysis

**Throughput Improvement**:
- **Absolute Gain**: 450,000 - 120,000 = 330,000 tokens/second
- **Relative Improvement**: (450,000 / 120,000) = 3.75× improvement
- **Per-GPU Efficiency**: 7,031 vs 7,500 tokens/second/GPU

**Latency Reduction**:
- **Absolute Reduction**: 8.3 - 2.2 = 6.1ms per token
- **Relative Improvement**: (8.3 / 2.2) = 3.77× faster

### 3.3 Scaling Analysis

**Linear Scaling Verification**:
- **Theoretical Scaling**: 4× GPUs (16→64) should give 4× improvement
- **Actual Scaling**: 3.75× improvement (94% efficiency)
- **Scaling Loss**: 6% due to communication overhead

**Resource Utilization**:
- **GPU Compute**: >95% utilization per expert
- **Network**: 60% of peak bandwidth utilized
- **Memory**: 85% of GPU memory utilized

### 3.4 Bottleneck Analysis

**Baseline Bottlenecks**:
- **Expert Contention**: 4 experts sharing single GPU
- **Pipeline Stalls**: Sequential processing through stages
- **Memory Bandwidth**: Shared among multiple experts
- **Compute Saturation**: Partial GPU utilization per expert

**Proposed Solution Benefits**:
- **No Contention**: One expert per GPU
- **Parallel Processing**: All experts compute simultaneously
- **Full Utilization**: Each GPU dedicated to single expert
- **Overlap Communication**: Hide network latency

## 4. Detailed Measurements

### 4.1 Communication Overhead

**Network Traffic**:
- **Token Transfer Volume**: 8192 bytes × 10.24M tokens = 80GB per batch
- **Per-GPU Traffic**: 80GB / 64 = 1.25GB per GPU
- **Transfer Time**: 1.25GB / 450GB/s = 2.8ms (overlapped with computation)

### 4.2 Memory Access Patterns

**Baseline Memory Usage**:
- **Expert Parameters**: 4 × 537MB = 2.15GB per GPU
- **Token Buffer**: 2.5GB shared across experts
- **Total**: ~5GB per GPU

**Proposed Memory Usage**:
- **Expert Parameters**: 537MB per GPU
- **Token Buffer**: 2.5GB per GPU
- **Communication Buffer**: 16MB per GPU
- **Total**: ~3GB per GPU

### 4.3 Load Balancing Effectiveness

**Expert Utilization Distribution**:
- **Mean**: 10.24M tokens / 64 experts = 160K tokens per expert
- **Standard Deviation**: ±5% (due to dynamic load balancing)
- **Maximum Deviation**: <10% from mean
- **Token Drop Rate**: <0.1%

## 5. Discussion

### 5.1 Architectural Implications

**Design Trade-offs**:
- **GPUs vs Performance**: 4× more GPUs for 3.75× performance gain
- **Cost Efficiency**: Higher absolute performance, lower GPU efficiency
- **Scalability**: Linear scaling up to 64 GPUs demonstrated

### 5.2 Network Requirements

**Minimum Requirements**:
- **Bandwidth**: 100 Gbps per GPU minimum
- **Latency**: <10μs for optimal overlap
- **Topology**: Fat-tree or torus interconnect preferred

### 5.3 Future Scaling

**Beyond 64 Experts**:
- **Theoretical**: Linear scaling to 128+ experts
- **Practical**: Network topology becomes critical
- **Limitations**: All-to-all communication complexity O(n²)

## 6. Validation Summary

**Key Findings**:
1. **Large EP (≥16) is effective**: Demonstrated 3.75× throughput improvement
2. **One-expert-per-GPU works**: Eliminates intra-GPU contention
3. **Communication overlap succeeds**: Hides network latency effectively
4. **Load balancing is crucial**: Prevents expert stragglers
5. **Linear scaling achieved**: 94% scaling efficiency demonstrated

**Validation of Claims**:
- ✅ Large EP ≥ 16 improves performance
- ✅ Cross-node distribution scales effectively
- ✅ Communication overlap mitigates network cost
- ✅ Load balancing prevents bottlenecks
- ✅ Near-linear scaling in HPC environment