# Experiments: Large-Scale Cross-Node Expert Parallelism

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Expert type**: Multi-Layer Perceptron (MLP)
- **Precision**: FP16 (half precision)
- **Batch size**: 1024 tokens per forward pass
- **Multi-Head Attention**: 16 heads, 512 dimensions per head
- **MLP hidden size**: 32,768 dimensions
- **Setting**: Inference-only

### 1.2 Hardware Environment
- **GPU**: H100 GPUs
- **Interconnect**: High-bandwidth (NVLink/InfiniBand/H100 NVSwitch)
- **Environment**: High-performance computing (HPC) cluster

### 1.3 Evaluation Metrics
- **TPS (Tokens per Second)**: Overall throughput measurement
- **TPOT (Time per Output Token)**: Per-token latency measurement

## 2. Parallel Deployment Configurations

### 2.1 Baseline Configuration (TP=8, PP=2)

#### 2.1.1 Resource Allocation
- **Total GPUs**: 16 H100 GPUs
- **Tensor Parallelism (TP)**: 8-way
- **Pipeline Parallelism (PP)**: 2 stages
- **Expert Parallelism (EP)**: Not explicitly used (experts colocated)

#### 2.1.2 Per-GPU Deployment
- **Tensor shards**: Each GPU holds 1/8 of tensor-parallel shard for all layers
- **Pipeline stages**: Each stage spans 8 GPUs (total 2 stages)
- **Expert placement**: 4 experts per GPU (colocated)
- **Processing flow**: Tokens flow sequentially through pipeline stages
- **Resource sharing**: Multiple experts share GPU compute resources

### 2.2 Proposed Cross-Node Expert Parallelism

#### 2.2.1 Resource Allocation
- **Total GPUs**: 64 H100 GPUs
- **Expert Parallelism (EP)**: 64-way (16 experts/layer × 4 layers = 64 unique experts)
- **Tensor Parallelism (TP)**: Optional TP=2 if single expert exceeds GPU memory
- **Pipeline Parallelism (PP)**: Each MoE layer as micro-stage

#### 2.2.2 Per-GPU Deployment
- **Expert placement**: Exactly one expert per GPU
- **Expert distribution**: 64 experts across 64 GPUs (1:1 mapping)
- **Layer-wise distribution**: 
  - Layer 1: Experts 1-16 on GPUs 1-16
  - Layer 2: Experts 17-32 on GPUs 17-32
  - Layer 3: Experts 33-48 on GPUs 33-48
  - Layer 4: Experts 49-64 on GPUs 49-64
- **Memory usage**: Each expert has dedicated GPU memory
- **Processing flow**: Tokens dynamically routed to expert's GPU

#### 2.2.3 Routing and Communication
- **Token routing**: Dynamic routing to GPU holding target expert
- **Communication pattern**: Cross-node token transfers
- **Overlap strategy**: Asynchronous token sending with computation
- **Batching**: Tokens grouped by destination expert

## 3. Experimental Results

### 3.1 Performance Comparison
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) | Improvement |
|--------|-----------|-------------------|----------------|-----------|-------------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 | - |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 | 3.75× TPS, 3.8× lower latency |

### 3.2 Detailed Analysis
- **Throughput improvement**: 450,000 vs 120,000 TPS (3.75× increase)
- **Latency reduction**: 2.2ms vs 8.3ms TPOT (3.8× decrease)
- **Resource utilization**: Full GPU utilization with dedicated expert per GPU
- **Scalability**: Near-linear scaling with 64 GPUs in large EP regime

### 3.3 Bottleneck Analysis
- **Baseline bottlenecks**:
  - Intra-GPU contention between 4 experts
  - Pipeline stalls between stages
  - Shared compute resources limiting parallelism
- **Proposed method advantages**:
  - No intra-GPU contention
  - Maximum expert-level parallelism
  - Overlapped communication hiding latency

## 4. Experimental Insights

### 4.1 Scaling Behavior
- **Linear scaling**: Achieved when EP ≥ 16 (large EP regime)
- **Network efficiency**: High-bandwidth interconnects enable cross-node communication
- **Load balancing**: Dynamic gating prevents expert overloading

### 4.2 Communication Overhead
- **Amortized cost**: Communication overhead spread across many tokens
- **Overlap effectiveness**: Async routing minimizes waiting time
- **Topology awareness**: Optimal expert placement reduces network congestion

### 4.3 Memory Considerations
- **Per-expert memory**: 32,768 hidden size × FP16 = 65,536 bytes per weight matrix
- **GPU memory**: Each H100 has sufficient memory for single expert
- **Optional TP**: TP=2 available if expert exceeds single-GPU capacity

## 5. Validation of Key Claims

### 5.1 Expert Independence
- **Validation**: Each expert processes tokens independently
- **Measurement**: No intra-GPU resource contention observed
- **Result**: Full compute utilization per expert

### 5.2 Communication Efficiency
- **Validation**: Cross-node communication effectively overlapped
- **Measurement**: <5% communication overhead in total latency
- **Result**: Near-linear scaling achieved

### 5.3 Load Balancing
- **Validation**: Dynamic gating maintains balanced expert loads
- **Measurement**: <10% variance in tokens per expert
- **Result**: No straggler experts observed

## 6. Reproducibility Details

### 6.1 Exact Configuration
- **Model**: 4-layer MoE, 16 experts/layer, MLP experts
- **Dimensions**: 16 heads × 512 dim/head = 8192 attention dim
- **Hidden**: 32,768 MLP hidden size
- **Precision**: FP16 throughout
- **Batch**: 1024 tokens fixed
- **GPUs**: H100 with identical specs

### 6.2 Measurement Protocol
- **TPS**: Measured over 1000 forward passes
- **TPOT**: Average of per-token latency across batch
- **Warmup**: 100 warmup iterations before measurement
- **Repetitions**: 5 runs averaged for final results