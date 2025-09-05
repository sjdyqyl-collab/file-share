# Phase 3: Experiments Extraction

## 1. Experimental Setup

### 1.1 Model Configuration
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per Layer**: 16 experts
- **Expert Type**: Each expert is a Multi-Layer Perceptron (MLP)
- **Precision**: FP16 (half precision)

### 1.2 Input Configuration
- **Batch Size**: 1024 sequences per batch
- **Sequence Length**: 10,000 tokens per sequence
- **Total Tokens per Batch**: 10,240,000 tokens (1024 × 10,000)

### 1.3 Model Dimensions
- **Multi-Head Attention (MHA)**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total attention dimension: 16 × 512 = 8,192
- **MLP Hidden Size**: 32,768 (for each expert)

### 1.4 Hardware Environment
- **GPU Type**: NVIDIA H100
- **Setting**: Inference-only (no training)
- **Network Infrastructure**: NVLink, InfiniBand, H100-class NVSwitch fabrics

## 2. Evaluation Metrics

### 2.1 Primary Metrics
- **TPS (Tokens per Second)**: Overall throughput measurement
- **TPOT (Time per Output Token)**: Latency per token in milliseconds

### 2.2 Performance Objectives
- Maximize TPS (higher is better)
- Minimize TPOT (lower is better)

## 3. Parallel Deployment Configurations

### 3.1 Baseline Configuration (Traditional Approach)
- **GPUs Used**: 16 H100 GPUs
- **Parallel Strategy**: 
  - Tensor Parallelism (TP): 8
  - Pipeline Parallelism (PP): 2
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (2 stages total)
  - Expert placement: 4 experts colocated per GPU
- **Processing Flow**: Tokens flow sequentially through pipeline stages with shared GPU resources among experts

### 3.2 Proposed Configuration (Cross-Node Expert Parallelism)
- **GPUs Used**: 64 H100 GPUs
- **Parallel Strategy**:
  - Expert Parallelism (EP): 64 (16 experts × 4 layers distributed)
  - Tensor Parallelism (TP): 1 (optional TP=2 if expert doesn't fit)
  - Pipeline Parallelism (PP): Micro-stages per MoE layer
- **Per-GPU Allocation**:
  - Each GPU hosts exactly one expert
  - No expert colocation on same GPU
  - Token communication overlapped with computation
- **Routing Strategy**:
  - Dynamic token routing to GPU holding corresponding expert
  - Asynchronous token batch sending
  - Minimal idle time through overlapping

## 4. Experimental Results

### 4.1 Performance Comparison Table
| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### 4.2 Performance Improvements
- **Throughput Improvement**: 450,000 ÷ 120,000 = 3.75× higher TPS
- **Latency Reduction**: 8.3 ÷ 2.2 = 3.77× lower TPOT (approximately 3.8× as reported)
- **GPU Utilization**: 4× more GPUs used (16 → 64) with 3.75× throughput gain indicates near-linear scaling

### 4.3 Resource Utilization Analysis
- **Baseline Issues**:
  - Intra-GPU contention from 4 experts sharing GPU resources
  - Pipeline stalls due to sequential processing
  - Limited expert-level parallelism
- **Proposed Benefits**:
  - Dedicated GPU resources per expert eliminate contention
  - Maximal expert-level parallelism (64 concurrent experts)
  - Efficient overlapping of communication and computation

## 5. Scalability Analysis

### 5.1 Scaling Characteristics
- **Large EP Regime**: EP=64 qualifies as large EP (≥16)
- **Near-linear Scaling**: 4× GPU increase with 3.75× throughput gain
- **Communication Overhead**: Successfully mitigated through asynchronous routing and overlapping

### 5.2 Bottleneck Analysis
- **Baseline**: Compute contention on shared GPUs
- **Proposed**: Network communication (effectively managed through topology-aware placement)

## 6. Experimental Validation

### 6.1 Test Environment
- **Inference-only Setting**: No training overhead
- **Stable Conditions**: Fixed batch size and sequence length
- **Controlled Variables**: Same model architecture, only parallelism strategy differs

### 6.2 Reproducibility Factors
- **Hardware Specification**: Clearly defined H100 cluster
- **Model Configuration**: Precise dimensions and precision
- **Parallel Strategies**: Explicit TP/PP/EP values for both configurations
- **Metrics**: Standard TPS and TPOT measurements

## 7. Experimental Conclusions

### 7.1 Key Findings
- One-expert-per-GPU deployment significantly outperforms traditional colocation
- Large EP (≥16) enables effective scaling in HPC environments
- Communication overhead can be effectively managed through careful scheduling

### 7.2 Validation of Hypotheses
- **Hypothesis**: Maximizing expert-level parallelism improves performance
- **Validation**: 3.75× throughput improvement achieved
- **Hypothesis**: Modern networks can sustain large EP communication
- **Validation**: Near-linear scaling observed with effective overlapping strategies