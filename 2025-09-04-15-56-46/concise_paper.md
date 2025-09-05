# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## **Abstract**

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## **Introduction**

Mixture-of-Experts (MoE) architectures enable scaling large language models while maintaining computational efficiency by activating only a subset of experts per input token. However, traditional MoE parallelization strategies assign multiple experts to the same GPU, creating computational bottlenecks and limiting expert-level parallelism as cluster sizes grow.

We present a cross-node expert parallelism method that distributes experts across nodes with at most one expert per GPU, pushing Expert Parallelism (EP) to 16 or beyond. This maximizes concurrent computation by leveraging modern HPC networking capabilities.

## **Methods**

### **1. Expert Placement Strategy**

#### **1.1 Single-Expert-Per-GPU Deployment**
- **Constraint**: Each GPU hosts at most one expert
- **Distribution**: For E experts and G GPUs, ensure each expert on distinct GPU if E ≤ G
- **Replication**: When E > G, replicate experts to maximize concurrency
- **Benefit**: Each expert processes tokens without intra-GPU contention

#### **1.2 Cross-Node Distribution**
- **Topology-aware placement** considering:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link

### **2. Routing and Load Balancing**

#### **2.1 Gating and Token Routing**
- **Top-K gating**: Standard MoE gating network (typically K=2)
- **Token batching**: Group tokens by destination expert
- **Asynchronous routing**: Send token batches while overlapping computation
- **Dynamic load balancing**: Monitor per-expert load and adjust gating probabilities

#### **2.2 Communication Overlap**
- **Interleaving**: Expert computation and token transfers
- **Implementation**: CUDA streams and asynchronous communication (NCCL/MPI)
- **Pipeline scheduling**: Process partial batches as they arrive

### **3. Scalability Framework**

#### **3.1 Large EP Regime (EP ≥ 16)**
- **Definition**: Expert Parallelism degree ≥ 16
- **Limiting factor**: Network bandwidth (mitigated by topology-aware routing)
- **Compute efficiency**: All GPUs fully utilized

#### **3.2 Integration with Other Parallelisms**
- **Tensor Parallelism (TP)**: Within expert if FFN exceeds GPU memory
- **Data Parallelism (DP)**: Across MoE network replicas
- **Synchronized updates**: Maintaining expert-level parallelism

## **Experiments**

### **1. Experimental Setup**

#### **1.1 Model Configuration**
- **Architecture**: 4-layer MoE model
- **Experts**: 16 experts per layer (64 total experts)
- **Expert type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16
- **Batch size**: 1024 tokens per forward pass

#### **1.2 Transformer Specifications**
- **Multi-Head Attention**: 16 heads × 512 dimensions per head
- **MLP hidden size**: 32,768

### **2. Baseline vs Proposed Comparison**

| **Configuration** | **GPUs** | **Per-GPU Deployment** | **TPS** | **TPOT** |
|------------------|----------|------------------------|---------|----------|
| **Baseline** (TP=8, PP=2) | 16 H100 | 4 experts + TP shard per GPU | 120,000 | 8.3 ms |
| **Proposed** (Large EP) | 64 H100 | 1 expert per GPU | 450,000 | 2.2 ms |

### **3. Results Analysis**
- **Throughput improvement**: 3.75× higher (450k vs 120k TPS)
- **Latency reduction**: 3.8× lower (2.2ms vs 8.3ms TPOT)
- **Scalability**: Near-linear scaling with 64 GPUs
- **Resource utilization**: Complete GPU utilization without expert contention

## **Conclusion**

Our large-scale cross-node expert parallelism method maximizes expert-level parallelism by deploying one expert per GPU, achieving significant performance improvements in high-performance computing environments. The approach shifts the computational bottleneck from intra-GPU contention to network communication, effectively mitigated through asynchronous token routing and topology-aware placement. Validated on a 4-layer, 64-expert MoE model, this method provides a scalable blueprint for future high-performance MoE inference deployments.

## **Key Technical Specifications**

### **Model Dimensions**
- **Layers**: 4
- **Experts per layer**: 16
- **Total experts**: 64
- **MHA heads**: 16
- **MHA head dimension**: 512
- **MLP hidden size**: 32,768
- **Precision**: FP16
- **Batch size**: 1024

### **Hardware Requirements**
- **GPU type**: H100
- **Network**: NVLink, InfiniBand, NVSwitch
- **Optimal configuration**: 64 GPUs for 64 experts (one expert per GPU)
- **Minimum EP regime**: 16 or more experts per parallel group