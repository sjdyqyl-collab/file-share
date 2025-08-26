## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Experiments**

### **1. Experimental Setup**
- **Model**: 4-layer Mixture-of-Experts (MoE)
- **Experts**: 16 experts per layer, each expert is a MLP
- **Precision**: FP16
- **Batch size**: 1024 tokens per forward pass
- **MHA Configuration**: 16 heads, 512 dimensions per head
- **MLP Hidden size**: 32768
- **Environment**: H100 GPUs, inference-only setting

**Metrics:**
- **TPS (Tokens per Second)**: Measures throughput
- **TPOT (Time per Output Token)**: Measures latency per token

### **2. Parallel Deployment Details**

#### **2.1 Baseline Deployment (TP=8, PP=2)**
- **GPUs Used**: 16 H100
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage (2 stages total) spans 8 GPUs
  - Experts are colocated on GPUs: 4 experts per GPU
- **Processing**: Tokens flow sequentially through pipeline stages with shared compute resources

#### **2.2 Proposed Cross-Node Expert Parallelism**
- **GPUs Used**: 64 H100 (one GPU per expert per layer)
- **Per-GPU Allocation**:
  - Each GPU hosts **exactly one expert**
  - Tensor parallelism: TP=2 only if single expert's FFN cannot fit on one GPU
  - Pipeline parallelism: Each MoE layer as a micro-stage
  - Communication: Token communication overlapped with computation
- **Routing**:
  - Input tokens dynamically routed to GPU holding corresponding expert
  - Token batches asynchronously sent to minimize idle time
- **Expert Distribution**: All 64 experts per layer compute in parallel

### **3. Results**

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

**Performance Improvements:**
- **Throughput**: 3.75× higher (450k vs 120k TPS)
- **Latency**: 3.8× lower (2.2ms vs 8.3ms TPOT)

### **4. Discussion**
- **Resource Utilization**: One expert per GPU enables full GPU compute and memory utilization
- **Communication Efficiency**: Asynchronous token routing ensures minimal waiting across nodes
- **Scalability**: Near-linear scaling achieved with 64 GPUs in large EP regime (EP ≥ 16)
- **Bottleneck Shift**: Successfully shifted from intra-GPU contention to manageable network communication
- **Practical Validation**: Demonstrates real-world effectiveness on H100 clusters with substantial performance gains

### **5. Experimental Configuration Details**
- **Network**: High-bandwidth interconnects (NVLink, InfiniBand)
- **Software**: NCCL/MPI for communication, CUDA streams for async operations
- **Load Balancing**: Runtime monitoring and dynamic gating adjustment
- **Memory Management**: Efficient token batching and pipeline scheduling
- **Reproducibility**: All experiments conducted in controlled inference environment with consistent batch sizes and model configurations