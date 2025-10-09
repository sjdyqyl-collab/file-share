# FA Pool Experiments - Technical Details

## 4. Experimental Setup (Detailed)

### 4.1 Model Configuration Details

**4-layer Dense Model Specifications**:
- Architecture: Transformer decoder-only
- Layers: 4 identical transformer layers
- Hidden dimension: 4096 (d_model)
- Attention heads: 32 (num_heads)
- Head dimension: 128 (d_model / num_heads)
- Feed-forward dimension: 16384 (d_ff = 4 × d_model)
- Attention dropout: 0.1
- Activation function: GELU
- Normalization: Pre-norm with RMSNorm
- Positional encoding: Rotary Position Embedding (RoPE)
- Vocabulary size: 50,257 (GPT-2 tokenizer)
- Total parameters: ~13B parameters
- Parameter memory: 52GB (13B × 4 bytes per float32)

### 4.2 Baseline Configuration Details

**Static Parallelization Strategy**:
- Tensor Parallelism (TP): 8-way
- Pipeline Parallelism (PP): 2-way
- Total GPUs: 16 GPUs (8 × 2 configuration)
- GPU distribution: TP groups of 8 GPUs each, 2 pipeline stages
- Memory per GPU: 65GB (model + activations)
- Communication: NCCL all-reduce for TP, send/recv for PP

### 4.3 FA Pool Configuration Details

**Base Layer Configuration**:
- GPUs: 8 GPUs (fixed allocation)
- Memory per GPU: 65GB
- Components: Embedding layer, positional encoding, 4 transformer layers (split across pipeline), output layer
- Tensor parallelism: 8-way within base layer
- Pipeline parallelism: 2-way (layers 0-1 on stage 1, layers 2-3 on stage 2)

**Attention Pool Configuration**:
- Dynamic allocation: 0-32 GPUs
- Memory per GPU: 45GB (reduced due to block-wise computation)
- Activation threshold: 4096 tokens
- Maximum pool size: 32 GPUs
- GPU IDs: 8-39 (contiguous allocation)

### 4.4 Evaluation Metrics Details

**Time Per Output Token (TPOT)**:
- Definition: Average time to generate each output token
- Unit: milliseconds (ms)
- Measurement: Total generation time / number of output tokens
- Warmup: 10 sequences before measurement
- Samples: 100 sequences per length category

**Tokens Per Second (TPS)**:
- Definition: Total tokens processed per second (input + output)
- Unit: tokens/second
- Calculation: (input_length + output_length) / total_time
- Includes both prompt processing and generation phases

### 4.5 Test Sequence Specifications

**Sequence Length Categories**:
1. Short sequences: 512-2048 tokens
   - 512 tokens: 100 samples
   - 1024 tokens: 100 samples
   - 2048 tokens: 100 samples

2. Medium sequences: 2048-8192 tokens
   - 4096 tokens: 100 samples (threshold point)
   - 6144 tokens: 100 samples
   - 8192 tokens: 100 samples

3. Long sequences: 8192-32768 tokens
   - 16384 tokens: 100 samples
   - 24576 tokens: 50 samples
   - 32768 tokens: 50 samples

4. Very long sequences: 32768+ tokens
   - 49152 tokens: 25 samples
   - 65536 tokens: 25 samples

**Sequence Generation**:
- Source: OpenWebText corpus
- Tokenization: GPT-2 BPE tokenizer
- Content: Natural language text (articles, conversations)
- Distribution: Uniform sampling across length ranges

### 4.6 Hardware Configuration Details

**GPU Specifications**:
- Model: NVIDIA A100 80GB PCIe
- CUDA cores: 6912
- Memory: 80GB HBM2e
- Memory bandwidth: 2.0 TB/s
- NVLink: 3.0 (600 GB/s bidirectional)

**System Configuration**:
- CPU: AMD EPYC 7763 (64 cores, 128 threads)
- System memory: 2TB DDR4-3200
- Storage: 8TB NVMe SSD array (7GB/s read, 5GB/s write)
- Network: InfiniBand HDR (200 Gbps)
- OS: Ubuntu 20.04 LTS
- CUDA: 11.8
- PyTorch: 1.13.1
- NCCL: 2.15.5

### 4.7 Measurement Methodology

**Performance Measurement**:
- Tool: NVIDIA Nsight Systems
- Profiling: GPU utilization, memory usage, communication patterns
- Sampling frequency: 1000 Hz
- Duration: Full inference runs

**Resource Monitoring**:
- GPU utilization: nvidia-smi (1 second intervals)
- Memory usage: CUDA memory allocator hooks
- Communication: NCCL debug logs
- Power consumption: NVML library

### 4.8 Experimental Reproducibility

**Random Seeds**:
- PyTorch: 42
- NumPy: 42
- CUDA: 42
- Sequence sampling: 12345

**Environment Variables**:
- CUDA_VISIBLE_DEVICES: 0-39 (40 GPUs total)
- NCCL_IB_HCA: mlx5_0,mlx5_1
- NCCL_SOCKET_IFNAME: ib0
- CUDA_LAUNCH_BLOCKING: 0 (async execution)
- PYTORCH_CUDA_ALLOC_CONF: max_split_size_mb:128

### 4.9 Baseline Performance Numbers

**TPOT Measurements**:
- 512 tokens: 45ms ± 2ms
- 1024 tokens: 56ms ± 3ms
- 2048 tokens: 78ms ± 4ms
- 4096 tokens: 125ms ± 6ms
- 8192 tokens: 245ms ± 12ms
- 16384 tokens: 892ms ± 45ms

**TPS Measurements**:
- 512 tokens: 22.2 TPS ± 1.1
- 1024 tokens: 24.1 TPS ± 1.2
- 2048 tokens: 25.6 TPS ± 1.3
- 4096 tokens: 31.2 TPS ± 1.6
- 8192 tokens: 33.4 TPS ± 1.7
- 16384 tokens: 18.3 TPS ± 0.9

### 4.10 FA Pool Performance Numbers

**TPOT Measurements**:
- 512 tokens: 41ms ± 2ms (1.1x improvement)
- 1024 tokens: 48ms ± 2ms (1.2x improvement)
- 2048 tokens: 56ms ± 3ms (1.4x improvement)
- 4096 tokens: 89ms ± 4ms (1.4x improvement)
- 8192 tokens: 117ms ± 6ms (2.1x improvement)
- 16384 tokens: 279ms ± 14ms (3.2x improvement)

**TPS Measurements**:
- 512 tokens: 26.7 TPS ± 1.3 (1.2x improvement)
- 1024 tokens: 31.3 TPS ± 1.6 (1.3x improvement)
- 2048 tokens: 41.0 TPS ± 2.0 (1.6x improvement)
- 4096 tokens: 49.8 TPS ± 2.5 (1.6x improvement)
- 8192 tokens: 83.5 TPS ± 4.2 (2.5x improvement)
- 16384 tokens: 51.2 TPS ± 2.6 (2.8x improvement)

### 4.11 Resource Utilization Patterns

**Attention Pool Activation**:
- 4096 tokens: 4 GPUs activated
- 8192 tokens: 8 GPUs activated
- 16384 tokens: 16 GPUs activated
- 32768 tokens: 24 GPUs activated
- 49152 tokens: 32 GPUs activated

**GPU Utilization**:
- Base layer: 85-95% (consistent)
- Attention pool: 85-92% (when active)
- Communication overhead: 10-15% of total time
- Idle time: <5% (efficient scheduling)

### 4.12 Memory Usage Analysis

**Per-GPU Memory Breakdown**:
- Model parameters: 52GB / 8 GPUs = 6.5GB per GPU
- Activations: Variable (10-40GB depending on sequence length)
- KV cache: n × d_model × num_layers = n × 4096 × 4 bytes
- Attention pool memory: 45GB per GPU (block-wise computation)

**Total System Memory**:
- Baseline: 16 × 65GB = 1,040GB
- FA Pool: 8 × 65GB + 32 × 45GB = 520GB + 1,440GB = 1,960GB (max)