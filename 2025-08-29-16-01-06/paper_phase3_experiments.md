# Phase Three: Experiments Extraction

## **1. Experimental Setup**

### **1.1 Model Configuration**
- **Architecture**: 4-layer Mixture-of-Experts (MoE)
- **Experts per layer**: 16 experts
- **Total experts**: 64 experts (4 layers × 16 experts/layer)
- **Expert type**: Multi-Layer Perceptron (MLP)
- **Precision**: FP16 (half precision)
- **Batch size**: 1024 tokens per forward pass
- **Multi-Head Attention (MHA)**: 
  - Number of heads: 16
  - Dimension per head: 512
- **MLP hidden size**: 32,768

### **1.2 Evaluation Metrics**
- **TPS (Tokens per Second)**: Measures overall throughput
- **TPOT (Time per Output Token)**: Measures latency per token (in milliseconds)

### **1.3 Environment**
- **Hardware**: H100 GPUs
- **Setting**: Inference-only (no training)
- **Network**: High-bandwidth interconnect (NVLink, InfiniBand)

## **2. Parallel Deployment Details**

### **2.1 Baseline Configuration (TP=8, PP=2)**
- **Total GPUs**: 16 H100 GPUs
- **Parallelism Strategy**:
  - **Tensor Parallelism (TP)**: 8-way
  - **Pipeline Parallelism (PP)**: 2 stages
- **Per-GPU Allocation**:
  - Each GPU holds 1/8 of the tensor-parallel shard for all layers
  - Each pipeline stage spans 8 GPUs (16 total GPUs / 2 stages = 8 GPUs per stage)
  - Experts are colocated on GPUs: **4 experts per GPU** (64 experts / 16 GPUs = 4 experts/GPU)
- **Processing Flow**:
  - Tokens flow sequentially through the 2 pipeline stages
  - Multiple experts per GPU share compute resources, causing contention

### **2.2 Proposed Cross-Node Expert Parallelism**
- **Total GPUs**: 64 H100 GPUs
- **Parallelism Strategy**:
  - **Expert Parallelism (EP)**: 64 (maximum possible)
  - **Tensor Parallelism (TP)**: 1 (default), 2 if expert exceeds GPU memory
  - **Pipeline Parallelism**: Each MoE layer as micro-stage
- **Per-GPU Allocation**:
  - **Each GPU hosts exactly one expert** (64 experts / 64 GPUs = 1 expert/GPU)
  - No tensor parallelism applied unless memory constraints require (TP=2)
  - Each expert has dedicated GPU resources
- **Routing Mechanism**:
  - Input tokens dynamically routed to GPU holding the corresponding expert
  - Token batches sent asynchronously to minimize idle time
  - Communication overlapped with computation

## **3. Experimental Results**

### **3.1 Performance Comparison Table**

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 4 experts + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 64 | 1 expert per GPU | 450,000 | 2.2 |

### **3.2 Performance Analysis**
- **Throughput Improvement**: 450,000 / 120,000 = **3.75× higher TPS**
- **Latency Reduction**: 8.3 / 2.2 = **3.77× lower TPOT** (approximately 3.8× as stated)
- **Resource Utilization**: Full utilization of all 64 GPUs vs 16 GPUs in baseline
- **Expert Contention**: Eliminated in proposed method (1 expert/GPU vs 4 experts/GPU)

### **3.3 Scalability Characteristics**
- **Linear Scaling**: Near-linear throughput scaling with additional GPUs
- **Communication Overhead**: Effectively mitigated through asynchronous routing and computation overlap
- **Memory Efficiency**: Single expert per GPU allows maximum memory utilization per expert

## **4. Discussion**

### **4.1 Key Findings**
- **Expert Isolation**: Deploying one expert per GPU eliminates intra-GPU contention
- **Communication Management**: Asynchronous token routing ensures minimal waiting even across nodes
- **Scalability**: With 64 GPUs, system achieves near-linear scaling in large EP regime (EP ≥ 16)

### **4.2 Bottleneck Analysis**
- **Baseline Bottlenecks**:
  - Intra-GPU contention from 4 experts sharing resources
  - Pipeline stalls from sequential processing
  - Limited parallelism (only 16 GPUs active)
- **Proposed Solution Benefits**:
  - No expert contention (1 expert per GPU)
  - Maximum parallel computation (64 experts simultaneously)
  - Overlapped communication hides network latency

### **4.3 Configuration Validation**
- **EP=64**: Achieves maximum expert parallelism with 64 GPUs
- **TP=1**: Default configuration, TP=2 only if memory constraints exist
- **Batch Size=1024**: Optimal for balancing throughput and latency
- **FP16**: Maintains precision while maximizing performance

## **5. Experimental Summary**
The experiments demonstrate that the proposed cross-node expert parallelism method achieves:
- **3.75× throughput improvement** over baseline
- **3.8× latency reduction** compared to traditional approaches
- **Full utilization** of 64 H100 GPUs with dedicated expert per GPU
- **Effective scalability** in the large EP regime (EP ≥ 16)
- **Practical deployment feasibility** for large-scale MoE inference workloads