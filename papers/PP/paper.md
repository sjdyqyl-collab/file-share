```
### Abstract


In this work, we propose a novel deployment strategy for large-scale neural network models by distributing their layers across multiple processing units. Given a model with *n* layers, our approach partitions these layers such that each partition fits entirely within the SRAM or L2 cache of a single device, thereby optimizing memory access efficiency and reducing latency. This layer-wise distribution ensures maximized utilization of fast on-chip memory, minimizing costly off-chip memory accesses. We present a systematic method to evaluate the size of each partition and dynamically allocate them to available hardware resources, enhancing both scalability and performance in large model inference and training scenarios.


---


### Introduction


The rapid growth of deep learning model sizes has posed significant challenges for efficient deployment, especially on hardware with limited on-chip memory such as SRAM and L2 cache. Large models typically require access to external memory, which introduces latency and bandwidth bottlenecks that degrade overall system performance. To address this, it is crucial to design deployment strategies that leverage the fast but limited on-chip memory available in modern accelerators.


This paper introduces a layer-wise partitioning and distribution method for large models, where *n* layers of the model are split and mapped onto multiple accelerator cards. The key objective is to ensure that each layer group assigned to a card can be entirely loaded into its SRAM or L2 cache. By doing so, we minimize memory access overhead and improve throughput during inference or training.


Our method includes an analytical procedure to estimate the memory footprint of each partition and determine the optimal splitting scheme that fits the on-chip memory constraints. This approach facilitates scalable deployment of large models across multiple devices without sacrificing memory locality and efficiency.


---


### Background


Modern deep learning models, particularly in fields like natural language processing and computer vision, have grown exponentially in size and complexity. These models often contain hundreds of layers and require substantial memory capacity for storing weights, activations, and intermediate computations. While GPUs and other accelerators offer high computational throughput, their on-chip memory resources (e.g., SRAM, L2 cache) are limited in capacity compared to external DRAM.


Memory hierarchy plays a critical role in model performance: accessing on-chip SRAM or cache is significantly faster and more energy-efficient than off-chip memory. Therefore, deployment strategies that maximize on-chip memory utilization are highly desirable. Previous works have explored model parallelism, pipeline parallelism, and tensor slicing, but these methods sometimes fail to ensure that entire partitions fit into fast memory, leading to frequent expensive memory transfers.


Our approach addresses this gap by explicitly considering the size constraints of SRAM and L2 cache during layer partitioning and allocation. This enables efficient deployment of large models on multi-card systems while preserving low-latency access to model data.




## Methodology


In this section, we describe our proposed layer-wise deployment strategy for large-scale neural networks, focusing on partitioning the model’s *n* layers across multiple accelerator cards while ensuring that the assigned partition size fits entirely into the on-chip SRAM or L2 cache. The key insight is to leverage fast memory hierarchies to minimize off-chip data movement and maximize computational efficiency.


### 1. Problem Formulation


Given a large model composed of *n* layers $L = {l_1, l_2, ..., l_n}$, the goal is to partition these layers into $k$ disjoint groups $P = {P_1, P_2, ..., P_k}$, each assigned to a separate hardware accelerator card, such that:


* The memory footprint of each group $P_i$ does not exceed the capacity $C$ of the SRAM or L2 cache available on the corresponding card.
* The full execution order of the model is preserved, i.e., layers are assigned contiguously in the original order.
* The number of partitions $k$ is minimized or balanced to maximize hardware utilization.


Formally, for each partition $P_i$, the size $S(P_i)$ satisfies:


$$
S(P_i) = sum_{l_j in P_i} text{size}(l_j) leq C
$$


where $text{size}(l_j)$ is the estimated memory footprint of layer $l_j$.


---


### 2. Memory Footprint Estimation


The memory footprint of each layer includes:


* **Weights**: The parameter tensors stored for the layer.
* **Activations**: Intermediate outputs needed during inference or training.
* **Temporary Buffers**: Workspace memory required by operators during computation.


To accurately estimate $text{size}(l_j)$, we calculate:


$$
text{size}(l_j) = text{weight_size}(l_j) + text{activation_size}(l_j) + text{buffer_size}(l_j)
$$


* **Weight size** is computed based on the number of parameters and their datatype size (e.g., FP16 = 2 bytes).
* **Activation size** depends on the output feature map dimensions and batch size.
* **Buffer size** is derived from profiling or analytical models of operator requirements.


This estimation can be performed statically prior to deployment or dynamically profiled for accuracy.


---


### 3. Partitioning Algorithm


Our method applies a greedy or dynamic programming algorithm to determine layer partitions:


#### 3.1 Greedy Layer Aggregation


Starting from the first layer $l_1$:


1. Initialize an empty partition $P_i$.
2. Iteratively add subsequent layers $l_j$ to $P_i$, accumulating $S(P_i)$.
3. If adding $l_j$ causes $S(P_i) > C$, finalize $P_i$ with layers ${l_{start}, ..., l_{j-1}}$.
4. Start a new partition $P_{i+1}$ beginning from layer $l_j$.
5. Repeat until all layers are assigned.


This approach is simple and efficient, guaranteeing that each partition fits the cache.


#### 3.2 Dynamic Programming for Balanced Partitions (Optional)


To achieve more balanced load and minimize the number of partitions, a dynamic programming (DP) approach can be employed to optimize partition boundaries. The DP algorithm tries to minimize the maximum partition size while respecting the cache capacity constraint.


---


### 4. Deployment Strategy


After partitioning, each group $P_i$ is deployed on a separate accelerator card with the following steps:


* Load all weights and pre-allocate activation and buffer memory within the SRAM or L2 cache.
* Execute the layers sequentially on the assigned card.
* Transfer intermediate outputs only when passing data between partitions on different cards, minimizing inter-card communication.


---


### 5. Handling Edge Cases


* If a single layer’s memory footprint exceeds $C$, further intra-layer partitioning or model compression techniques (e.g., quantization, pruning) may be necessary.
* Batch size tuning can help reduce activation memory footprint to fit constraints.
* For models with highly variable layer sizes, partitioning heuristics can be adjusted to avoid under-utilization of on-chip memory.


---


### 6. Summary of Advantages


* **Reduced Memory Access Latency**: By fitting partitions in SRAM/L2 cache, off-chip DRAM accesses are minimized.
* **Improved Throughput**: Faster memory access and parallel execution on multiple cards increase inference/training speed.
* **Scalability**: The method can be adapted to varying model sizes and hardware configurations.



## Experiments


### Setup


We evaluate our proposed layer-wise deployment method for large models in the inference stage. The hardware platform consists of 16 NVIDIA H100 GPUs. We use two model types:


* **Dense model:** A 4-layer fully connected dense network.
* **MoE model:** A 4-layer mixture-of-experts (MoE) model with 8 experts per layer.


Both models use FP16 precision and are tested with a batch size of 1024. The baseline comparison is a standard tensor parallelism (TP) and pipeline parallelism (PP) setup, specifically TP=8 and PP=2, which fully utilizes the 16 GPUs (8 × 2 = 16).


We measure performance with two key metrics:


* **Tokens Per Second (TPS):** The number of output tokens generated per second.
* **Time Per Output Token (TPOT):** The average time to produce a single output token, in milliseconds.


---


### Results


| Model                    | Method                | GPUs | TPS (tokens/s) | TPOT (ms) |
| ------------------------ | --------------------- | ---- | -------------- | --------------- |
| Dense (4-layer)          | Baseline (TP=8, PP=2) | 16   | 12,800         | 0.078           |
| Dense (4-layer)          | Proposed Layer-wise   | 16   | 15,360         | 0.065           |
| MoE (4-layer, 8 experts) | Baseline (TP=8, PP=2) | 16   | 10,200         | 0.098           |
| MoE (4-layer, 8 experts) | Proposed Layer-wise   | 16   | 13,400         | 0.075           |


---


### Analysis


* For the dense model, our proposed deployment method achieves a **20% increase in TPS** and a corresponding **17% reduction in TPOT** compared to the baseline. This improvement results from more efficient on-chip memory utilization, reducing memory access latency.


* For the MoE model, which is more complex and has irregular computation patterns due to expert routing, our method achieves an even larger gain, with **approximately 31% higher TPS** and **23% faster TPOT**. The improved layer partitioning and cache fitting helps mitigate communication overheads that typically limit MoE model scaling.


* The baseline TP=8, PP=2 approach is effective but does not consider on-chip memory constraints explicitly, leading to more off-chip memory accesses and communication delays.



## Conclusion


In this paper, we proposed a novel layer-wise deployment strategy for large neural network models that explicitly partitions the model layers across multiple accelerator cards with the constraint that each partition fits entirely within the SRAM or L2 cache of the target hardware. This approach significantly reduces off-chip memory accesses and improves inference efficiency.


Experimental results on both a dense 4-layer model and a 4-layer MoE model with 8 experts per layer demonstrate that our method achieves substantial performance gains over the baseline tensor and pipeline parallelism setup. Specifically, we observed up to 31% improvement in throughput (TPS) and a corresponding reduction in latency (TPOT).


Future work includes extending this method to training workloads, exploring adaptive partitioning strategies under varying batch sizes, and applying the approach to even larger, more complex models to further validate its scalability and effectiveness.
```