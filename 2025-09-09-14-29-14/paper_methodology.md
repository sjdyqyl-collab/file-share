# Methodology Extraction - Large-Scale Cross-Node Expert Parallelism

## 1. Expert Placement Strategy

### 1.1 Single-Expert-Per-GPU Deployment
**Core Principle**: Each GPU hosts at most one expert
- **Mathematical Constraint**: For E experts and G GPUs, ensure each expert is assigned to a distinct GPU if E ≤ G
- **Replication Strategy**: If E > G, experts are replicated across GPUs to maximize concurrent independent experts while balancing memory usage
- **Resource Utilization**: Each expert processes tokens without contention from other experts on the same device

### 1.2 Cross-Node Distribution Algorithm
**Topology-Aware Placement Strategy**:
- **Input Parameters**:
  - Node-to-node bandwidth matrix: B[i,j] for nodes i and j
  - GPU memory capacity per node: M[node_id]
  - Expected token routing patterns: P[token, expert]
- **Optimization Objective**: Minimize max(tokens_sent_across_any_single_link)
- **Constraint**: Maintain one-expert-per-GPU principle
- **Placement Algorithm**:
  ```
  1. Initialize expert_locations = {}
  2. For each expert e in E:
     - Select GPU g with minimal communication cost
     - Ensure g has sufficient memory for expert e
     - Assign expert_locations[e] = g
     - Update GPU memory usage
  3. Validate load balancing across nodes
  ```

## 2. Routing and Load Balancing

### 2.1 Gating Mechanism
**Standard MoE Routing**:
- **Top-K Selection**: For each input token, select top-K experts based on gating scores
- **Gating Network**: Standard learned gating mechanism determining expert activation

### 2.2 Token Sharding Across Nodes
**Cross-Node Token Transfer Protocol**:
1. **Token Batching**:
   - Group tokens by destination expert
   - Reduce network messages through batching
   - Batch size optimization: min(batch_size, network_bandwidth / latency)

2. **Asynchronous Routing**:
   - Send token batches asynchronously while overlapping expert computation
   - Use CUDA streams or NCCL for non-blocking transfers
   - Communication-computation overlap ratio: ≥ 0.8

3. **Load Balancing**:
   - Monitor per-expert load: L[e] = tokens_processed[e] / time_window
   - Dynamic gating adjustment: Adjust gating probabilities to prevent overloading
   - Load balancing threshold: |L[e] - mean(L)| / mean(L) < 0.2

## 3. Communication Overlap and Scheduling

### 3.1 Overlapping Compute and Communication
**Implementation Details**:
- **CUDA Streams**: Utilize separate streams for computation and communication
- **NCCL/MPI**: Leverage asynchronous communication libraries
- **Overlap Strategy**: While batch n is processed, transfer batch n+1 simultaneously
- **Synchronization Points**: Ensure data dependencies are maintained

### 3.2 Pipeline Scheduling
**Multi-Layer MoE Network Scheduling**:
- **Micro-Stage Definition**: Each MoE layer treated as a micro-stage
- **Immediate Routing**: Token outputs from layer i immediately routed to layer i+1
- **Partial Batch Processing**: Experts start processing as soon as partial batch arrives
- **Pipeline Depth**: Number of layers = 4 (as per experimental setup)

## 4. Scalability Considerations

### 4.1 Large EP Regime (EP ≥ 16)
**Optimization Focus**:
- **Network Bandwidth**: Primary limiting factor in large EP regime
- **Topology-Aware Routing**: Minimize communication across slow links
- **Token Batching**: Amortize communication overhead across many tokens
- **Compute Saturation**: Ensure all GPUs are fully utilized for compute

### 4.2 Memory and Model Parallelism Integration
**Hybrid Parallelism Strategy**:
- **Tensor Parallelism (TP)**: Applied within single expert if FFN cannot fit on one GPU
  - TP degree: 2 (optional, only when needed)
  - Matrix partitioning: Column-parallel for first linear, row-parallel for second linear
- **Data Parallelism (DP)**: Applied across replicas of MoE network
  - DP degree: Determined by total GPUs / (EP × TP × PP)
  - Synchronized weight updates maintaining expert-level parallelism

## 5. Model Architecture Details

### 5.1 MoE Layer Configuration
- **Number of Layers**: 4
- **Experts per Layer**: 16
- **Expert Type**: MLP (Multi-Layer Perceptron)
- **Precision**: FP16

### 5.2 Model Dimensions
- **Token Dimension**: 8192
- **Hidden Size of MLP**: 32768
- **MHA Configuration**:
  - Number of heads: 16
  - Dimension per head: 512
  - Total MHA dimension: 16 × 512 = 8192

### 5.3 Batch Configuration
- **Batch Size**: 1024 sequences
- **Sequence Length**: 10000 tokens per sequence
- **Total Tokens per Batch**: 1024 × 10000 = 10,240,000 tokens

## 6. Implementation Constraints

### 6.1 Deployment Constraints
- **Maximum Experts per GPU**: 1
- **Minimum EP Degree**: 16 (large EP regime)
- **GPU Memory per Expert**: Must fit MLP with hidden size 32768
- **Network Requirements**: High-bandwidth interconnects (NVLink, InfiniBand, H100 NVSwitch)

### 6.2 Communication Requirements
- **Token Transfer Size**: 8192 dimensions × 2 bytes (FP16) = 16,384 bytes per token
- **Peak Communication**: 10,240,000 tokens × 16,384 bytes = 167.8 GB per batch
- **Required Bandwidth**: ≥ 400 GB/s for 450,000 TPS throughput

## 7. Algorithm Summary

### 7.1 Expert Placement Algorithm
```
function place_experts(E, G, topology):
    expert_map = {}
    gpu_memory = initialize_gpu_memory(G)
    
    for expert in E:
        best_gpu = None
        min_cost = infinity
        
        for gpu in G:
            if gpu_memory[gpu] >= expert.memory_required:
                cost = calculate_communication_cost(expert, gpu, topology)
                if cost < min_cost:
                    min_cost = cost
                    best_gpu = gpu
        
        expert_map[expert] = best_gpu
        gpu_memory[best_gpu] -= expert.memory_required
    
    return expert_map
```

### 7.2 Token Routing Algorithm
```
function route_tokens(tokens, expert_map, gating_network):
    expert_batches = defaultdict(list)
    
    for token in tokens:
        top_experts = gating_network.top_k(token, k=2)
        for expert in top_experts:
            gpu_id = expert_map[expert]
            expert_batches[gpu_id].append(token)
    
    return expert_batches
```