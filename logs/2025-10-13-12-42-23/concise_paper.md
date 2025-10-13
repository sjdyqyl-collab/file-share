# Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models

## Abstract

We propose a large-scale cross-node expert parallelism strategy for Mixture-of-Experts (MoE) models, designed to maximize computational parallelism by deploying at most one expert per GPU. Unlike conventional approaches that colocate multiple experts on the same device, our method fully exploits distributed resources to reduce expert-level contention and improve throughput. By ensuring that Expert Parallelism (EP) is at least 16—qualifying as "large EP" in our definition—we significantly increase the independence of expert computation, enabling better scalability and reduced inter-expert interference. This approach is particularly effective in high-performance computing (HPC) and large GPU cluster environments, where the balance between communication overhead and compute saturation is critical.

## Introduction

Mixture-of-Experts (MoE) architectures have emerged as a powerful approach for scaling large language models (LLMs) while maintaining computational efficiency. By activating only a subset of experts per input token, MoE models can achieve higher parameter counts without proportionally increasing the inference or training cost. However, scaling MoE models across large GPU clusters introduces significant challenges in expert placement and parallelization.

Traditional MoE parallelization strategies often assign multiple experts to the same GPU to reduce inter-node communication. While this minimizes network traffic, it also creates computational bottlenecks and limits the degree of true expert parallelism. As model and cluster sizes grow, this trade-off becomes increasingly suboptimal.

In this work, we present a cross-node expert parallelism method that prioritizes distributing experts across nodes such that each GPU hosts at most one expert. By pushing Expert Parallelism (EP) to 16 or beyond, we unlock higher degrees of concurrent computation, allowing each expert to run in near isolation. This design shifts the optimization focus from reducing communication to maximizing compute concurrency, leveraging modern HPC networking capabilities to sustain high bandwidth and low latency across nodes.

## Methods

### 1. Expert Placement Strategy

#### 1.1 Single-Expert-Per-GPU Deployment
Our method deploys at most one expert per GPU to maximize expert-level parallelism. For a MoE layer with E experts and a cluster of G GPUs, we ensure that each expert is assigned to a distinct GPU if E ≤ G. If E > G, we replicate experts across GPUs in a manner that maximizes the concurrency of independent experts while balancing memory usage. This approach ensures that each expert can process tokens without contention from other experts on the same device, fully utilizing GPU compute units.

#### 1.2 Cross-Node Distribution
Experts are distributed across nodes using a topology-aware placement strategy that considers node-to-node bandwidth and latency, GPU memory capacity per node, and expected token routing patterns. The placement algorithm aims to minimize the maximum number of tokens sent across any single link while maintaining the one-expert-per-GPU principle.

### 2. Routing and Load Balancing

#### 2.1 Gating Mechanism
The routing of tokens to experts is governed by a gating network where top-K gating scores determine which subset of experts is activated for each token.

#### 2.2 Token Sharding Across Nodes
Our approach includes: (1) Token Batching - Group tokens by destination expert to reduce the number of network messages; (2) Asynchronous Routing - Send token batches asynchronously to overlapping expert computation; (3) Load Balancing - Monitor per-expert load and dynamically adjust gating probabilities to avoid overloading specific experts.

### 3. Communication Overlap and Scheduling

#### 3.1 Overlapping Compute and Communication
We interleave expert computation and communication where one batch of tokens is processed on a GPU while the next batch is simultaneously transferred from other nodes. CUDA streams or asynchronous communication libraries (e.g., NCCL or MPI) are leveraged to ensure that data transfer does not block GPU computation.

#### 3.2 Pipeline Scheduling
In multi-layer MoE networks, token outputs from the previous MoE layer are immediately routed to the next layer's experts, and experts in subsequent layers start processing as soon as a partial batch arrives, rather than waiting for the full batch.

### 4. Scalability Considerations

#### 4.1 Large EP Regime (EP ≥ 16)
Our method is optimized for large EP setups where network bandwidth becomes the primary limiting factor. We mitigate this through topology-aware routing and token batching. The one-expert-per-GPU policy ensures that all GPUs are fully utilized for compute while communication costs are amortized across many tokens.

#### 4.2 Memory and Model Parallelism Integration
To handle very large models that cannot fit on a single GPU, each expert can be further partitioned using tensor model parallelism (TP) within its GPU if necessary. Data parallelism (DP) is applied across replicas of the MoE network, allowing synchronized weight updates while maintaining high expert-level parallelism.

## Experiments

### 1. Experimental Setup

We evaluate the proposed method in an inference-only setting using H100 GPUs:
- **Model**: 4-layer Mixture-of-Experts (MoE), 16 experts per layer, each expert is a MLP
- **Precision**: FP16
- **Batch size**: 1024 sequences
- **Sequence Length**: 10,000 tokens per sequence
- **Token Dimension**: 8,192
- **MHA**: 16 heads, 512 dimensions per head
- **MLP Hidden Size**: 32,768

**Metrics**: TPS (Tokens per Second) and TPOT (Time per Output Token)

### 2. Parallel Deployment Details

#### 2.1 Baseline Deployment (TP=8, PP=2)
- **GPUs Used**: 16 H100
- **Per-GPU Allocation**: Each GPU holds 1/8 of the tensor-parallel shard for all layers; Each pipeline stage (2 stages total) spans 8 GPUs; Experts are colocated on GPUs, typically 8 experts each layer per GPU
- **Processing**: Tokens flow sequentially through the pipeline stages, and multiple experts per GPU share compute resources

#### 2.2 Proposed Cross-Node Expert Parallelism
- **GPUs Used**: 16 H100 (one GPU per expert per layer)
- **Per-GPU Allocation**: Each GPU hosts exactly one expert per layer
- **Routing**: Input tokens are dynamically routed to the GPU holding the corresponding expert; Token batches are asynchronously sent, ensuring minimal idle time

### 3. Results

| Method | GPUs Used | Per-GPU Deployment | TPS (Tokens/s) | TPOT (ms) |
|--------|-----------|-------------------|----------------|-----------|
| Baseline (TP=8, PP=2) | 16 | 8 experts each layer + TP shard per GPU | 120,000 | 8.3 |
| Proposed Cross-Node Expert Parallelism | 16 | 1 expert each layer per GPU | 450,000 | 2.2 |

**Performance Analysis**:
- Throughput improvement: 3.75× higher (450,000 vs 120,000 TPS)
- Latency reduction: 3.8× lower (2.2ms vs 8.3ms TPOT)
- Full GPU utilization with dedicated expert per GPU
- No intra-GPU contention between experts

### 4. Discussion

Deploying one expert per GPU allows full utilization of GPU compute and memory. Asynchronous token routing ensures minimal waiting, even across nodes. With 16 GPUs, the system scales near-linearly in the large EP regime (EP ≥ 16).

## Conclusion

We proposed a large-scale cross-node expert parallelism method for Mixture-of-Experts (MoE) models, designed to maximize expert-level parallelism by deploying at most one expert per GPU. Our approach shifts the computational bottleneck from intra-GPU contention to communication, which is effectively mitigated through asynchronous token routing, topology-aware expert placement, and overlap of computation with communication.

We demonstrated the effectiveness in an inference-only setting on a 4-layer, 16-expert-per-layer MoE model using FP16 precision and a batch size of 1024. Compared to a baseline configuration with TP=8 and PP=2, our approach achieved ~3.75× higher throughput and ~3.8× lower latency by fully utilizing all 16 GPUs and enabling large Expert Parallelism (EP ≥ 16).

Our method provides a scalable blueprint for future high-performance MoE inference, particularly in environments with abundant GPU resources such as H100 clusters.

## Key Technical Specifications

### Model Dimensions
- Layers: 4
- Experts per Layer: 16
- Token Dimension: 8192
- MLP Hidden Size: 32768
- Precision: FP16

### Parallelism Parameters
- Large EP Regime: EP ≥ 16
- Expert Distribution: 1 expert per GPU
- Total GPUs: 16
- Memory per GPU: 8000 MB per expert

### Performance Metrics
- Baseline TPS: 120,000
- Proposed TPS: 450,000
- Speedup: 3.75×
- Latency Reduction: 3.8×