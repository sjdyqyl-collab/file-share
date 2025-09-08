# Large-Scale Cross-Node Expert Parallelism - Experiments

## **1. Experimental Setup**

### **1.1 Model Configuration**
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts (baseline) to 64 experts (proposed)
- **Expert type**: MLP-based feed-forward network
- **Precision**: FP16 (half-precision floating point)
- **Batch size**: 1024 sequences per batch
- **Sequence length**: 10,000 tokens per sequence
- **Token dimension**: 8192 dimensions per token
- **Multi-head attention**: 16 heads, 512 dimensions per head
- **MLP hidden size**: 32,768 hidden units

### **1.2 Hardware Environment**
- **GPU type**: NVIDIA H100 GPUs
- **GPU memory**: H100-class memory capacity
- **Network interconnect**: NVLink, InfiniBand, H100-class NVSwitch
- **Scale**: 16 GPUs (baseline) vs 64 GPUs (proposed)

### **1.3 Evaluation Metrics**
- **TPS (Tokens per Second)**: Overall throughput measurement
- **TPOT (Time per Output Token)**: Latency per token in milliseconds

## **2. Baseline Deployment (TP=8, PP=2)**

### **2.1 Configuration Details**
- **Total GPUs**: 16 H100 GPUs
- **Tensor Parallelism (TP)**: 8-way tensor parallelism
- **Pipeline Parallelism (PP)**: 2 pipeline stages
- **Expert Parallelism (EP)**: Limited by colocation

### **2.2 GPU Allocation**
- **Per-GPU allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (16 GPUs / 2 stages)
  - Experts are colocated on GPUs: typically 4 experts per GPU
- **Processing flow**: Tokens flow sequentially through pipeline stages
- **Resource contention**: Multiple experts per GPU share compute resources

### **2.3 Performance Results**
- **TPS**: 120,000 tokens per second
- **TPOT**: 8.3 milliseconds per token

## **3. Proposed Cross-Node Expert Parallelism**

### **3.1 Configuration Details**
- **Total GPUs**: 64 H100 GPUs
- **Expert Parallelism (EP)**: 64-way expert parallelism (one expert per GPU)
- **Tensor Parallelism (TP)**: Optional TP=2 for very large experts
- **Pipeline Parallelism (PP)**: Each MoE layer as a micro-stage

### **3.2 GPU Allocation**
- **Per-GPU allocation**:
  - Each GPU hosts exactly one expert
  - Tensor parallelism applied only if single expert's FFN exceeds GPU memory
  - Pipeline stages: each MoE layer operates as independent micro-stage
- **Expert placement**: 64 experts per layer distributed across 64 GPUs
- **Communication**: Cross-node token routing with asynchronous transfers

### **3.3 Routing Strategy**
- **Dynamic routing**: Input tokens routed to GPU holding corresponding expert
- **Asynchronous transfers**: Token batches sent asynchronously to minimize idle time
- **Load balancing**: Runtime monitoring and adjustment of token distribution

### **3.4 Performance Results**
- **TPS**: 450,000 tokens per second
- **TPOT**: 2.2 milliseconds per token

## **4. Performance Comparison**

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) | Improvement |
|--------|-----------|-------------------|----------------|-----------|-------------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 | - |
| Proposed Cross-Node EP | 64 | 1 expert per GPU | 450,000 | 2.2 | 3.75× TPS, 3.8× TPOT |

## **5. Detailed Analysis**

### **5.1 Throughput Analysis**
- **Baseline bottleneck**: Intra-GPU contention from multiple experts sharing resources
- **Proposed advantage**: Full GPU utilization per expert with no resource sharing
- **Scaling factor**: 3.75× throughput improvement from 4× GPU increase (near-linear scaling)

### **5.2 Latency Analysis**
- **Baseline latency sources**: Pipeline stalls, expert contention, sequential processing
- **Proposed latency reduction**: Parallel expert processing, asynchronous communication overlap
- **Latency improvement**: 3.8× reduction in TPOT (8.3ms → 2.2ms)

### **5.3 Resource Utilization**
- **GPU compute**: 100% utilization per expert in proposed method vs shared utilization in baseline
- **Memory bandwidth**: Dedicated memory per expert vs shared memory in baseline
- **Network utilization**: Optimized cross-node communication vs minimal communication in baseline

## **6. Scalability Validation**

### **6.1 Large EP Regime Verification**
- **EP=64**: Validated large expert parallelism (≥16) effectiveness
- **Network impact**: Communication overhead successfully mitigated through overlapping
- **Compute saturation**: All 64 GPUs fully utilized for expert computation

### **6.2 Linear Scaling Demonstration**
- **GPU scaling**: 4× GPU increase (16→64) yields 3.75× throughput improvement
- **Efficiency**: 93.75% scaling efficiency (3.75/4.0)
- **Future scaling**: Method scales to even larger GPU counts

## **7. Experimental Constraints**

### **7.1 Inference-Only Setting**
- **Scope**: Experiments conducted in inference-only mode
- **Training extension**: Future work to explore training scenarios
- **Gradient considerations**: Not applicable in current inference setup

### **7.2 Model Size Limitations**
- **Expert size**: Each expert fits on single H100 GPU
- **TP usage**: Optional TP=2 for larger experts not extensively tested
- **Memory constraints**: Current setup within H100 memory limits

## **8. Reproducibility Details**

### **8.1 Environment Specifications**
- **CUDA version**: Compatible with H100 GPUs
- **Communication library**: NCCL or MPI for asynchronous transfers
- **Precision settings**: FP16 throughout experiments

### **8.2 Measurement Methodology**
- **TPS calculation**: Total tokens processed / total time
- **TPOT measurement**: End-to-end latency per output token
- **Warmup**: Sufficient warmup iterations to stabilize measurements