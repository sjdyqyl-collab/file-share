# Phase Two: Methodology Extraction

## **Method Overview**
The proposed method maximizes expert-level parallelism in large-scale MoE models through three key components:
1. Expert Placement Strategy - Assigning experts across GPUs and nodes
2. Routing and Load Balancing - Ensuring balanced input distribution to experts
3. Communication Overlap and Scheduling - Minimizing cross-node data transfer impact

## **1. Expert Placement Strategy**

### **1.1 Single-Expert-Per-GPU Deployment**
- **Constraint**: Deploy at most one expert per GPU
- **Implementation**: 
  - For E experts and G GPUs: assign each expert to distinct GPU if E ≤ G
  - If E > G: replicate experts across GPUs while maximizing concurrency and balancing memory
- **Benefit**: Eliminates intra-GPU contention, fully utilizes GPU compute units

### **1.2 Cross-Node Distribution**
- **Topology-aware placement** considers:
  - Node-to-node bandwidth and latency
  - GPU memory capacity per node
  - Expected token routing patterns
- **Objective**: Minimize maximum tokens sent across any single link while maintaining one-expert-per-GPU

## **2. Routing and Load Balancing**

### **2.1 Gating Mechanism**
- Standard MoE gating network determines expert activation
- Top-K gating scores select subset of experts per token

### **2.2 Token Sharding Across Nodes**
- **Token Batching**: Group tokens by destination expert to reduce network messages
- **Asynchronous Routing**: Send token batches asynchronously to overlap with expert computation
- **Load Balancing**: 
  - Monitor per-expert load
  - Dynamically adjust gating probabilities to prevent expert overload

## **3. Communication Overlap and Scheduling**

### **3.1 Overlapping Compute and Communication**
- **Interleaving Strategy**:
  - Process one token batch on GPU while transferring next batch from other nodes
  - Use CUDA streams or asynchronous libraries (NCCL/MPI) to prevent blocking

### **3.2 Pipeline Scheduling**
- **Multi-layer MoE networks**:
  - Route token outputs from previous layer immediately to next layer's experts
  - Start processing in subsequent layers as soon as partial batch arrives
- **Benefit**: Increases throughput, reduces expert idle time

## **4. Scalability Considerations**

### **4.1 Large EP Regime (EP ≥ 16)**
- **Definition**: Expert Parallelism degree ≥ 16 qualifies as "large EP"
- **Characteristics**:
  - Network bandwidth becomes primary limiting factor
  - One-expert-per-GPU ensures full GPU utilization
  - Communication costs amortized across many tokens

### **4.2 Memory and Model Parallelism Integration**
- **Large Model Handling**:
  - Partition experts using tensor model parallelism (TP) within GPU if needed
  - Apply data parallelism (DP) across MoE network replicas
- **Integration**: Maintains high expert-level parallelism while handling memory constraints

## **5. Implementation Details**

### **5.1 Hardware Requirements**
- **GPUs**: H100-class with NVLink/InfiniBand interconnect
- **Network**: High-bandwidth, low-latency fabric (H100 NVSwitch)
- **Memory**: Sufficient per-GPU memory for single expert

### **5.2 Software Stack**
- **Communication Libraries**: NCCL, MPI for asynchronous transfers
- **Scheduling**: CUDA streams for computation-communication overlap
- **Load Balancing**: Dynamic gating probability adjustment

### **5.3 Configuration Parameters**
- **Expert Parallelism**: EP = 64 (maximum possible with 64 experts)
- **Tensor Parallelism**: TP = 1 (default), TP = 2 if expert exceeds GPU memory
- **Pipeline Parallelism**: Each MoE layer as micro-stage
- **Batch Size**: 1024 tokens per forward pass
- **Precision**: FP16 for all computations

## **6. Method Summary**
The methodology achieves maximum expert parallelism through:
- **Spatial Distribution**: One expert per GPU across nodes
- **Temporal Overlap**: Asynchronous communication with computation
- **Load Optimization**: Dynamic routing and load balancing
- **Scalability Design**: Optimized for EP ≥ 16 regime with near-linear scaling

This approach fundamentally shifts the optimization focus from communication reduction to compute concurrency maximization, leveraging modern HPC networking capabilities.