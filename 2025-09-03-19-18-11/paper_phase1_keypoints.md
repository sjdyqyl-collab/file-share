# Paper Keypoints - Ring Attention + Sequence Parallelism

## Abstract (Original)
We present a novel parallelization strategy for Multi-Head Attention (MHA) in large-scale transformer models that combines Ring Attention with sequence parallelism. Our approach leverages the communication-efficient properties of the ring topology to distribute attention computation across devices, while sequence parallelism reduces memory footprint by splitting input sequences across workers. This design minimizes all-to-all communication overhead, enhances scalability for extremely long sequences, and enables efficient utilization of distributed hardware resources. Experimental analysis indicates that the proposed method achieves substantial throughput improvements compared to conventional data- and tensor-parallel approaches, particularly in scenarios with high sequence length and large model size.

## Key Contributions
1. **Novel Parallelization Strategy**: Combines Ring Attention with sequence parallelism for MHA in transformers
2. **Communication Efficiency**: Uses ring topology to reduce peak communication bandwidth vs all-to-all patterns
3. **Memory Optimization**: Sequence parallelism reduces activation memory by factor of P (number of devices)
4. **Scalability**: Particularly effective for long sequences (L > 16k tokens) and large models

## Core Problem Addressed
- Transformers' quadratic attention complexity and memory requirements create bottlenecks for distributed training/inference
- Multi-Head Attention requires large intermediate tensors and significant inter-device communication
- Traditional approaches struggle with extremely long sequences and large model sizes

## Technical Innovation
- **Ring Attention**: Devices arranged in logical ring, passing partial K/V blocks sequentially
- **Sequence Parallelism**: Splits sequence dimension L across P devices, each storing L/P tokens
- **Combined Approach**: Ring communication pattern + sequence partitioning = reduced memory + efficient communication

## Performance Gains
- **Dense Transformer (4L)**: 20.8% TPS improvement, 17.6% TPOT reduction
- **Tested on**: 16×H100 GPUs with NVLink/NVSwitch
- **Settings**: FP16, batch size 1024 tokens, 16 heads × 512 dim/head, MLP hidden size 32768

## Implementation Details
- Uses NCCL send/recv or MPI point-to-point
- Overlaps computation with asynchronous communication
- Mixed precision (fp16/bf16) for reduced bandwidth
- Fused kernels for projection and softmax
- Scales well with increasing L and P