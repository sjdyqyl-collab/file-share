# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **1. Introduction**

Mixture-of-Experts (MoE) architectures scale large language models efficiently by activating only a subset of experts per input token. However, scaling MoE models across large GPU clusters introduces challenges in expert placement and parallelization. Traditional strategies colocate multiple experts on the same GPU to reduce communication, creating computational bottlenecks and limiting expert-level parallelism.

We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU. By pushing Expert Parallelism (EP) to 16 or beyond, we unlock higher degrees of concurrent computation, leveraging modern HPC networking capabilities to sustain high bandwidth and low latency.

## **2. Methods**

### **2.1 Expert Placement Strategy**
- **Single-Expert-Per-GPU**: Deploy at most one expert per GPU
- **Assignment Rules**:
  - If E ≤ G: Each expert assigned to distinct GPU
  - If E > G: Replicate experts while maximizing concurrency
- **Cross-Node Distribution**: Topology-aware placement considering bandwidth, latency, GPU memory, and routing patterns

### **2.2 Routing and Load Balancing**
- **Gating**: Standard top-K gating mechanism
- **Token Sharding**:
  - Token batching by destination expert
  - Asynchronous routing overlapping computation
  - Dynamic load balancing through gating probability adjustment

### **2.3 Communication Overlap**
- **Compute-Communication Overlap**: Interleave expert computation with cross-node token transfers
- **Pipeline Scheduling**: Immediate routing between MoE layers with partial batch processing
- **Implementation**: CUDA streams, NCCL/MPI for asynchronous operations

### **2.4 Scalability Framework**
- **Large EP Regime**: EP ≥ 16
- **Memory Integration**: Tensor parallelism within GPU if needed, data parallelism across replicas
- **Focus**: Maximize compute concurrency, manage communication through scheduling

## **3. Experiments**

### **3.1 Setup**
- **Model**: 4-layer MoE, 16 experts per layer, MLP experts
- **Precision**: FP16
- **Input**: 1024 sequences × 10,000 tokens = 10.24M tokens/batch
- **Dimensions**: 16 heads × 512 dim/head = 8,192 attention dim, 32,768 MLP hidden size
- **Hardware**: H100 GPUs, inference-only

### **3.2 Configurations**

#### **Baseline (Traditional)**
- **GPUs**: 16 H100
- **Parallelism**: TP=8, PP=2
- **Deployment**: 4 experts + 1/8 tensor shard per GPU
- **Processing**: Sequential pipeline with shared GPU resources

#### **Proposed (Cross-Node Expert Parallelism)**
- **GPUs**: 64 H100
- **Parallelism**: EP=64, 1 expert per GPU
- **Deployment**: Each GPU hosts exactly one expert
- **Processing**: Asynchronous token routing with computation overlap

### **3.3 Results**
| Method | GPUs | Deployment | TPS | TPOT |
|--------|------|------------|-----|------|
| Baseline | 16 | 4 experts/GPU | 120,000 | 8.3ms |
| Proposed | 64 | 1 expert/GPU | 450,000 | 2.2ms |

**Improvements**: 3.75× higher throughput, 3.8× lower latency

## **4. Conclusion**

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU. In a 4-layer, 64-expert-per-layer MoE model, we achieved 3.75× higher throughput and 3.8× lower latency compared to traditional approaches, demonstrating effective scaling in the large EP regime (EP ≥ 16). This provides a scalable blueprint for high-performance MoE inference in GPU-rich environments.