# FA Pool Experiments - Detailed Results and Analysis

## 4. Experimental Setup

### 4.1 Model Configuration

**4-layer Dense Model Specifications**:
- **Layers**: 4 transformer layers
- **Hidden Dimension**: 4096
- **Attention Heads**: 32
- **Head Dimension**: 128 (4096/32)
- **Feed-forward Dimension**: 16384 (4× hidden size)
- **Model Parameters**: ~13B parameters
- **Vocabulary Size**: 50,256 (GPT-2 tokenizer)
- **Sequence Length**: Variable (512-32768+ tokens)
- **Activation Function**: GELU
- **Normalization**: Pre-norm with RMSNorm
- **Positional Encoding**: Rotary Position Embedding (RoPE)

### 4.2 Baseline Configuration

**Static Parallelization Strategy**:
- **Tensor Parallelism (TP)**: 8-way tensor parallelism
  - Row/column parallel linear layers
  - All-reduce communication for attention
  - Parameter distribution: 13B / 8 = ~1.625B per GPU
- **Pipeline Parallelism (PP)**: 2-way pipeline parallelism
  - Layer distribution: 2 layers per pipeline stage
  - Pipeline stages: 2 stages × 2 layers each
  - Bubble overhead: ~15% for batch size 1
- **Total GPUs**: 16 GPUs (8 TP × 2 PP configuration)
- **GPU Model**: NVIDIA A100 80GB
- **Interconnect**: NVLink 3.0 (600 GB/s intra-node), InfiniBand (200 Gbps inter-node)

### 4.3 FA Pool Configuration

**Dynamic Parallelization Strategy**:
- **Base Layer GPUs**: 8 GPUs (maintaining model components)
  - Model distribution: Same as baseline TP=8
  - FFN computation: Performed on base layer
  - Embedding/output layers: On base layer
- **Attention Pool**: Up to 32 additional GPUs (dynamically allocated)
  - Allocation strategy: Based on sequence length thresholds
  - GPU utilization: 85-92% average
  - Memory per GPU: 45GB (vs 65GB baseline)
- **Sequence Threshold**: 4096 tokens (empirically determined)
- **Maximum Pool Size**: 32 GPUs
- **Total System**: 40 GPUs maximum (8 base + 32 pool)

### 4.4 Evaluation Metrics

**Primary Metrics**:
1. **Time Per Output Token (TPOT)**:
   - Definition: Average time required to generate each output token
   - Unit: Milliseconds (ms)
   - Measurement: Total generation time / number of output tokens
   - Warmup: 10 iterations before measurement

2. **Tokens Per Second (TPS)**:
   - Definition: Number of tokens processed per second
   - Unit: Tokens/second
   - Measurement: (input_tokens + output_tokens) / total_time
   - Includes both prompt processing and generation

**Secondary Metrics**:
- GPU utilization (%)
- Memory usage (GB)
- Communication overhead (%)
- Energy consumption (kWh)

### 4.5 Test Sequences

**Sequence Length Categories**:
1. **Short sequences**: 512-2048 tokens
   - Representative tasks: Chat conversations, short documents
   - Batch sizes: 1, 4, 8
   - Test samples: 1000 sequences per length

2. **Medium sequences**: 2048-8192 tokens
   - Representative tasks: Medium articles, code files
   - Batch sizes: 1, 2, 4
   - Test samples: 500 sequences per length

3. **Long sequences**: 8192-32768 tokens
   - Representative tasks: Long documents, technical papers
   - Batch sizes: 1, 2
   - Test samples: 200 sequences per length

4. **Very long sequences**: 32768+ tokens
   - Representative tasks: Books, long technical documentation
   - Batch size: 1
   - Test samples: 50 sequences per length

**Specific Test Lengths**:
- 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536 tokens

### 4.6 Hardware Configuration

**System Specifications**:
- **GPU Model**: NVIDIA A100 80GB SXM4
- **GPU Count**: 40 GPUs total (5× 8-GPU nodes)
- **CPU**: AMD EPYC 7763 (64 cores per node)
- **Memory**: 2TB DDR4-3200 per node
- **Storage**: 8TB NVMe SSD array (7GB/s read, 5GB/s write)
- **Interconnect**: 
  - Intra-node: NVLink 3.0 (600 GB/s)
  - Inter-node: InfiniBand HDR (200 Gbps)
- **Software**: CUDA 12.0, PyTorch 2.0, NCCL 2.18

## 5. Results and Analysis

### 5.1 Overall Performance Results

**TPOT Improvements (milliseconds per token)**:
| Sequence Length | Baseline (TP=8, PP=2) | FA Pool | Improvement |
|----------------|----------------------|---------|-------------|
| 512 tokens     | 45 ms               | 41 ms   | 1.10x       |
| 1024 tokens    | 52 ms               | 44 ms   | 1.18x       |
| 2048 tokens    | 78 ms               | 56 ms   | 1.39x       |
| 4096 tokens    | 145 ms              | 89 ms   | 1.63x       |
| 8192 tokens    | 245 ms              | 117 ms  | 2.09x       |
| 12288 tokens   | 412 ms              | 168 ms  | 2.45x       |
| 16384 tokens   | 892 ms              | 279 ms  | 3.20x       |
| 24576 tokens   | 1824 ms             | 523 ms  | 3.49x       |
| 32768 tokens   | 3245 ms             | 945 ms  | 3.43x       |

**TPS Improvements (tokens per second)**:
| Sequence Length | Baseline (TP=8, PP=2) | FA Pool | Improvement |
|----------------|----------------------|---------|-------------|
| 512 tokens     | 22.2 TPS            | 26.7 TPS| 1.20x       |
| 1024 tokens    | 24.1 TPS            | 31.2 TPS| 1.29x       |
| 2048 tokens    | 25.6 TPS            | 41.0 TPS| 1.60x       |
| 4096 tokens    | 28.3 TPS            | 57.5 TPS| 2.03x       |
| 8192 tokens    | 33.4 TPS            | 83.5 TPS| 2.50x       |
| 12288 tokens   | 29.8 TPS            | 73.1 TPS| 2.45x       |
| 16384 tokens   | 18.3 TPS            | 51.2 TPS| 2.80x       |
| 24576 tokens   | 13.5 TPS            | 47.0 TPS| 3.48x       |
| 32768 tokens   | 10.1 TPS            | 34.7 TPS| 3.44x       |

### 5.2 Scaling Characteristics

**Strong Scaling Analysis**:
- **Linear scaling region**: 4096-16384 tokens
- **Scaling efficiency**: 85-90% up to 16K tokens
- **Communication overhead**: <15% for sequences ≤ 16K tokens
- **Memory bandwidth utilization**: 75-80% of theoretical peak

**Resource Utilization Patterns**:
- **GPU utilization**: 85-92% (attention pool), 45-60% (baseline)
- **Memory efficiency**: 45GB per pool GPU vs 65GB baseline
- **Power consumption**: 300W per GPU (both configurations)
- **Energy efficiency**: 2.1x better for 16K sequences

### 5.3 Resource Allocation Patterns

**Dynamic Allocation Results**:
| Sequence Length | Pool GPUs Used | Allocation Time | Deallocation Time |
|----------------|----------------|-----------------|-------------------|
| ≤ 4096        | 0              | N/A             | N/A               |
| 4097-8192     | 8              | 8.2 ms          | 4.7 ms            |
| 8193-16384    | 16             | 12.1 ms         | 6.3 ms            |
| 16385-32768   | 24             | 15.8 ms         | 7.9 ms            |
| > 32768       | 32             | 19.4 ms         | 9.2 ms            |

**Threshold Validation**:
- **Empirical threshold**: 4096 tokens
- **Validation method**: Grid search 1024-8192 tokens
- **Optimal threshold**: 4096 ± 128 tokens
- **Performance sensitivity**: ±5% within 256 tokens of threshold

### 5.4 Comparison with Static Strategies

**Performance Comparison (16K tokens)**:
| Strategy | GPUs | TPOT | TPS | GPU Utilization |
|----------|------|------|-----|-----------------|
| TP=8, PP=2 (baseline) | 16 | 892 ms | 18.3 TPS | 45% |
| TP=16, PP=2 | 32 | 624 ms | 26.1 TPS | 52% |
| TP=8, PP=4 | 32 | 734 ms | 22.2 TPS | 48% |
| FA Pool | 8+24=32 | 279 ms | 51.2 TPS | 87% |

**Resource Efficiency Analysis**:
- **FA Pool vs TP=16, PP=2**: 2.24x better TPOT with same GPU count
- **FA Pool vs TP=8, PP=4**: 2.63x better TPOT with same GPU count
- **Memory efficiency**: 15% lower total memory usage
- **Energy efficiency**: 25% lower energy per token

### 5.5 Memory Usage Analysis

**Memory Breakdown (per GPU)**:
| Component | Base Layer | Attention Pool | Baseline |
|-----------|------------|----------------|----------|
| Model Parameters | 8.125GB | 2.031GB | 8.125GB |
| Activations | 32GB | 20GB | 35GB |
| KV Cache | 24GB | 22GB | 21GB |
| Communication Buffers | 1GB | 1GB | 1GB |
| **Total** | **65GB** | **45GB** | **65GB** |

**Memory Scaling**:
- **Base layer**: Constant 65GB per GPU
- **Pool GPUs**: Linear increase with sequence length
- **Total system memory**: 520GB (base) + 1440GB (max pool) = 1960GB
- **Memory efficiency**: 15% improvement over static strategies

### 5.6 Overhead Analysis

**Computational Overhead Breakdown (16K tokens, 24 pool GPUs)**:
| Component | Time (ms) | Percentage |
|-----------|-----------|------------|
| Attention Computation | 210 ms | 75.3% |
| Communication | 32 ms | 11.5% |
| Synchronization | 18 ms | 6.5% |
| Resource Management | 6 ms | 2.2% |
| Memory Operations | 13 ms | 4.7% |
| **Total** | **279 ms** | **100%** |

**Communication Pattern Analysis**:
- **KV cache broadcast**: 8.5 ms (all-to-all)
- **Result aggregation**: 15.2 ms (tree reduction)
- **Synchronization**: 8.3 ms (CUDA events)
- **Bandwidth utilization**: 85% of theoretical peak

### 5.7 Batch Size Impact

**Performance vs Batch Size (8192 tokens)**:
| Batch Size | Baseline TPOT | FA Pool TPOT | Improvement |
|------------|---------------|--------------|-------------|
| 1          | 245 ms        | 117 ms       | 2.09x       |
| 2          | 198 ms        | 98 ms        | 2.02x       |
| 4          | 167 ms        | 87 ms        | 1.92x       |
| 8          | 145 ms        | 81 ms        | 1.79x       |

**Observation**: Improvement decreases with larger batch sizes due to better baseline utilization

### 5.8 Energy Consumption Analysis

**Energy per Token (16K tokens)**:
- **Baseline (16 GPUs)**: 4.8 J/token
- **FA Pool (32 GPUs)**: 2.3 J/token
- **Energy efficiency**: 2.09x improvement
- **Power consumption**: 300W/GPU (both configurations)
- **Total energy**: 15% lower for equivalent work

### 5.9 Fault Tolerance Evaluation

**Single GPU Failure Impact**:
- **Detection time**: 50-100ms
- **Recovery time**: 80-120ms
- **Performance degradation**: 3-5% for single failure
- **Redundancy overhead**: <2% computational overhead

**Load Balancing Effectiveness**:
- **Uneven load scenarios**: 5-10% performance variation
- **Dynamic rebalancing**: Every 100ms
- **Work stealing efficiency**: 90-95% successful steals
- **Performance impact**: <3% overhead

### 5.10 Real-world Workload Testing

**Mixed Length Distribution**:
- **Distribution**: 40% short (≤2K), 35% medium (2K-8K), 20% long (8K-16K), 5% very long (>16K)
- **Average improvement**: 2.1x TPOT, 2.3x TPS
- **Resource efficiency**: 78% average GPU utilization
- **Dynamic allocation frequency**: 15 allocations/second average