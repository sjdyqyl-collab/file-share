# Phase 2: Methodology Extraction

## Abstract (Retained)
We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Detailed Methodology

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Deployment
- **Principle**: At most one expert per GPU to eliminate intra-GPU contention
- **Mathematical Formulation**: For E experts and G GPUs, ensure each expert on distinct GPU when E ≤ G
- **Replication Strategy**: When E > G, replicate experts across GPUs maximizing independent expert concurrency
- **Memory Optimization**: Balance memory usage across GPUs during expert replication

#### 1.2 Cross-Node Distribution Algorithm
- **Topology Awareness**: Consider node-to-node bandwidth, latency, GPU memory capacity
- **Placement Optimization**: Minimize maximum tokens sent across any single link
- **Load Distribution**: Prevent hotspotting on any single node
- **Routing Pattern Prediction**: Account for expected token routing patterns in placement

### 2. Routing and Load Balancing Mechanism

#### 2.1 Gating Network Structure
- **Top-K Selection**: Standard MoE gating mechanism selecting top-K experts per token
- **Gating Scores**: Determine expert activation based on learned routing probabilities
- **Dynamic Adjustment**: Monitor per-expert load and adjust gating probabilities

#### 2.2 Token Sharding Strategy
- **Token Batching**: Group tokens by destination expert to minimize network messages
- **Batch Size Optimization**: Balance between communication efficiency and computation granularity
- **Asynchronous Routing**: Send token batches asynchronously while overlapping expert computation
- **Load Balancing Algorithm**: 
  - Monitor expert utilization in real-time
  - Dynamically adjust gating probabilities to prevent overloading
  - Ensure balanced workload distribution across all experts

### 3. Communication Overlap and Scheduling

#### 3.1 Compute-Communication Overlap
- **Dual Stream Architecture**: 
  - Compute stream for expert processing
  - Communication stream for token transfers
- **Batch Pipeline**: 
  - Process current batch while transferring next batch
  - Ensure continuous GPU utilization
- **CUDA Streams**: Utilize multiple CUDA streams for concurrent operations
- **NCCL/MPI Integration**: Leverage high-performance communication libraries

#### 3.2 Pipeline Scheduling Algorithm
- **Micro-Stage Design**: Each MoE layer as independent micro-stage
- **Token Flow**: Immediate routing from layer output to next layer's experts
- **Partial Batch Processing**: Start processing as soon as partial batch arrives
- **Dependency Management**: Ensure correct token ordering across pipeline stages

### 4. Memory and Model Parallelism Integration

#### 4.1 Tensor Model Parallelism (TP) within Expert
- **Application Condition**: When single expert FFN exceeds GPU memory
- **TP Degree**: Optional TP=2 for large experts
- **Partitioning Strategy**: 
  - Column-parallel for first linear layer
  - Row-parallel for second linear layer
- **Communication Overhead**: Minimize through expert-local TP

#### 4.2 Data Parallelism (DP) Integration
- **Replica Synchronization**: Synchronized weight updates across MoE replicas
- **Gradient Accumulation**: Maintain expert-level parallelism during training
- **Load Balancing Across Replicas**: Ensure balanced expert utilization across DP replicas

### 5. Large EP Regime Optimization (EP ≥ 16)

#### 5.1 Network Bandwidth Management
- **Primary Bottleneck Identification**: Network bandwidth in large EP regime
- **Topology-Aware Routing**: Minimize cross-node traffic through intelligent placement
- **Token Batching Optimization**: Maximize bandwidth utilization through batching

#### 5.2 Scalability Parameters
- **EP Range**: EP ≥ 16 qualifies as large EP
- **GPU Utilization**: Ensure all GPUs compute-constrained rather than communication-constrained
- **Linear Scaling Target**: Achieve near-linear scaling through optimal overlap

### 6. Implementation Details

#### 6.1 Hardware Requirements
- **GPU Type**: H100-class GPUs with high-bandwidth interconnects
- **Network Infrastructure**: NVLink, InfiniBand, or H100-class NVSwitch
- **Memory Requirements**: Sufficient per-GPU memory for single expert + activations

#### 6.2 Software Stack
- **Communication Libraries**: NCCL for GPU-GPU communication, MPI for multi-node
- **Scheduling Framework**: Custom scheduler for token routing and expert placement
- **Monitoring**: Real-time expert utilization and network traffic monitoring

#### 6.3 Configuration Parameters
- **Expert Count**: 16-64 experts per layer (configurable)
- **GPU Count**: Minimum 16 GPUs for large EP, optimal 64 GPUs
- **Batch Processing**: 1024 sequences per batch, 10,000 tokens per sequence
- **Precision**: FP16 for computation and communication
- **Token Dimension**: 8192-dimensional token representations