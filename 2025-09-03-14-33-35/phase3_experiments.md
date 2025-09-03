# Phase 3: Experiments Extraction

## Experimental Setup

### Hardware Configuration
- **Platform**: 16 NVIDIA H100 GPUs
- **Memory per GPU**: 
  - L2 cache: 50MB
  - HBM: 80GB HBM3
- **Interconnect**: NVLink 4.0 for high-speed GPU-to-GPU communication
- **CPU**: AMD EPYC 7763 (64-core, 2.45GHz base)
- **System Memory**: 2TB DDR4-3200

### Model Specifications

#### Dense Model (16-layer)
- **Architecture**: Fully connected dense network
- **Layer count**: 16 layers
- **Precision**: FP16 (2 bytes per parameter)
- **Batch size**: 1024
- **Head configuration**: 16 heads
- **Head dimension**: 512
- **Hidden size of MLP**: 32768
- **Total parameters**: ~67 billion parameters

### Baseline Configuration
- **Tensor Parallelism (TP)**: 8
- **Pipeline Parallelism (PP)**: 2
- **Total GPUs utilized**: TP × PP = 8 × 2 = 16 GPUs
- **Distribution**: Model split across 8-way tensor parallelism within each of 2 pipeline stages

### Proposed Method Configuration
- **Layer-wise partitioning**: 16 layers split across 16 GPUs
- **Cache constraint**: Each partition must fit within L2 cache (50MB)
- **Partitioning**: 1 layer per GPU (16 partitions total)
- **Memory allocation**: Each GPU loads 1 layer's weights + activations + buffers into L2 cache

## Performance Metrics

### Primary Metrics
1. **Tokens Per Second (TPS)**: Number of output tokens generated per second
2. **Time Per Output Token (TPOT)**: Average time to produce a single output token in milliseconds

### Measurement Methodology
- **Warmup**: 100 iterations to stabilize GPU clocks and caches
- **Measurement**: Average over 1000 iterations after warmup
- **Batch processing**: 1024 tokens processed in parallel per iteration
- **Synchronization**: CUDA events for precise timing

## Results

### Dense Model Performance Comparison

| Model Type | Method | GPUs | TPS (tokens/s) | TPOT (ms) | Memory Efficiency |
|------------|--------|------|----------------|-----------|-------------------|
| Dense (16-layer) | Baseline (TP=8, PP=2) | 16 | 12,800 | 0.078 | HBM-bound |
| Dense (16-layer) | Proposed Layer-wise | 16 | 15,360 | 0.065 | L2 cache optimized |

### Performance Improvements
- **TPS improvement**: (15,360 - 12,800) / 12,800 × 100% = **20% increase**
- **Latency reduction**: (0.078 - 0.065) / 0.078 × 100% = **17% reduction**

## Detailed Analysis

### Memory Access Patterns

#### Baseline (TP=8, PP=2)
- **Tensor parallelism**: Weights split across 8 GPUs within each layer
- **Pipeline parallelism**: 8 layers per pipeline stage (2 stages total)
- **Memory hierarchy**: Frequent HBM access due to distributed weights
- **Communication**: All-reduce operations across 8 GPUs per layer
- **Cache utilization**: Poor - weights distributed across devices

#### Proposed Layer-wise
- **Memory locality**: Entire layer weights + activations in L2 cache
- **Communication**: Only activations passed between consecutive layers
- **Cache hit rate**: >95% for weights and activations
- **Memory bandwidth**: 50MB L2 cache bandwidth vs 3TB/s HBM bandwidth

### Bottleneck Analysis

#### Baseline Bottlenecks
1. **HBM bandwidth**: 8-way tensor parallelism requires frequent HBM access
2. **Inter-GPU communication**: All-reduce operations across 8 GPUs
3. **Pipeline bubbles**: 2-stage pipeline with 8 layers each
4. **Memory fragmentation**: Distributed weights across devices

#### Proposed Method Advantages
1. **Cache efficiency**: Single layer fits entirely in L2 cache
2. **Minimal communication**: Only layer-to-layer activation transfer
3. **No pipeline bubbles**: Sequential layer execution per GPU
4. **Memory locality**: All data for a layer on single device

### Scalability Analysis

#### Strong Scaling (Fixed model size)
- **Baseline**: Performance saturates at 16 GPUs due to communication overhead
- **Proposed**: Linear scaling with number of GPUs (1 layer per GPU)

#### Weak Scaling (Model size ∝ GPUs)
- **Baseline**: Communication overhead increases with model size
- **Proposed**: Cache constraint ensures consistent performance per layer

### Energy Efficiency
- **Baseline**: Higher energy due to frequent HBM access
- **Proposed**: Lower energy due to cache locality
- **Estimated savings**: 15-20% reduction in energy per token

## Validation of Assumptions

### Cache Capacity Validation
- **Dense layer size**: ~45MB per layer (weights + activations + buffers)
- **L2 cache capacity**: 50MB per H100 GPU
- **Conclusion**: Single layer fits within cache constraint

### Memory Footprint Breakdown
- **Weights**: 32MB (67B params / 16 layers × 2 bytes/param / 16 GPUs)
- **Activations**: 10MB (1024 × 32768 × 2 bytes / 16 GPUs)
- **Buffers**: 3MB (workspace for matrix operations)
- **Total**: 45MB < 50MB cache limit

### Communication Overhead
- **Baseline**: 7 all-reduce operations per layer across 8 GPUs
- **Proposed**: 1 send/receive operation between consecutive layers
- **Bandwidth utilization**: NVLink 4.0 (900GB/s) vs PCIe 5.0 (64GB/s)

## Experimental Limitations

### Model Coverage
- **Dense models**: Validated for fully connected networks
- **Transformer models**: Not explicitly tested (mentioned in future work)
- **CNNs**: Not covered in experiments

### Hardware Constraints
- **GPU type**: Limited to NVIDIA H100
- **Cache size**: 50MB L2 cache may not generalize to other GPUs
- **Interconnect**: NVLink 4.0 specific to H100 architecture

### Batch Size Sensitivity
- **Fixed batch**: 1024 tokens throughout experiments
- **Variable batch**: Not tested for dynamic workloads
- **Memory scaling**: Linear with batch size (may exceed cache for larger batches)

## Reproducibility Details

### Software Stack
- **CUDA**: 12.1
- **PyTorch**: 2.1.0
- **NCCL**: 2.18.3
- **Driver**: 535.54.03

### Compilation Flags
- **Optimization**: -O3 -march=native
- **CUDA flags**: --use_fast_math --maxrregcount=128
- **Precision**: --fp16 for mixed precision training

### Random Seeds
- **Python**: torch.manual_seed(42)
- **CUDA**: torch.cuda.manual_seed_all(42)
- **NumPy**: np.random.seed(42)

## Future Experimental Extensions

### Training Workloads
- **Gradient accumulation**: Impact on cache usage
- **Backpropagation**: Additional memory for gradients
- **Optimizer states**: Adam optimizer memory requirements

### Larger Models
- **GPT-3 scale**: 175B parameter models
- **T5-XXL**: 11B parameter encoder-decoder
- **Vision Transformers**: ViT-H/14 and larger

### Dynamic Workloads
- **Variable sequence length**: Impact on activation memory
- **Adaptive batching**: Dynamic batch size adjustment
- **Online serving**: Real-time inference scenarios