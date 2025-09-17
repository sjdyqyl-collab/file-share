# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models (Concise Version)

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Traditional MoE parallelization assigns multiple experts per GPU to reduce communication, creating computational bottlenecks as models scale. We present a cross-node expert parallelism method distributing experts such that each GPU hosts at most one expert. By pushing EP ≥ 16, we unlock higher concurrent computation, shifting optimization from reducing communication to maximizing compute concurrency using modern HPC networking.

## **Methods**

### **1. Expert Placement Strategy**

#### **1.1 Single-Expert-Per-GPU Deployment**
- **Constraint**: At most one expert per GPU
- **Assignment**: E ≤ G → distinct GPU per expert; E > G → replicate maximizing concurrency
- **Benefit**: Eliminates intra-GPU expert contention

#### **1.2 Cross-Node Distribution**
- **Topology-aware placement** considering:
  - Node-to-node bandwidth/latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize max(tokens per link) while maintaining one-expert-per-GPU

### **2. Routing and Load Balancing**

#### **2.1 Token Sharding**
- **Batching**: Group tokens by destination expert
- **Asynchronous routing**: Overlap communication with computation
- **Dynamic balancing**: Monitor load, adjust gating probabilities

#### **2.2 Communication Overlap**
- **Interleaving**: Process batch i while transferring batch i+1
- **Technology**: CUDA streams, NCCL/MPI
- **Pipeline**: Immediate routing between layers, partial batch processing

### **3. Scalability Framework**

#### **3.1 Large EP Regime (EP ≥ 16)**
- **Network bandwidth**: Primary limiting factor
- **GPU utilization**: 100% compute via one-expert-per-GPU
- **Integration**: TP within experts (if needed), DP across replicas

## **Experiments**

### **1. Setup**
- **Model**: 4-layer MoE, 16 experts/layer (MLP), FP16
- **Input**: 1024 sequences × 10,000 tokens × 8192-dim
- **MHA**: 16 heads × 512-dim/head = 8192-dim total
- **MLP**: 32,768 hidden size per expert
- **Hardware**: 16× H100 GPUs

### **2. Configurations**

| Method | GPUs | Parallelism | Expert Placement | TPS | TPOT |
|--------|------|-------------|------------------|-----|------|
| Baseline | 16 | TP=8, PP=2 | 8 experts/GPU + TP shard | 120,000 | 8.3ms |
| Proposed | 16 | EP=16 | 1 expert/GPU | 450,000 | 2.2ms |
| **Improvement** | - | - | - | **3.75×** | **3.77×** |

### **3. Results**
- **Throughput**: 3.75× higher (450k vs 120k tokens/s)
- **Latency**: 3.77× lower (2.2ms vs 8.3ms TPOT)
- **GPU efficiency**: 28,125 vs 7,500 tokens/s/GPU

## **Conclusion**

Large-scale cross-node expert parallelism with one-expert-per-GPU deployment achieves 3.75× throughput and 3.77× latency improvements over traditional approaches by maximizing expert-level parallelism and overlapping communication with computation in EP ≥ 16 regimes.