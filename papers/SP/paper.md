```
**Abstract**
We present a novel parallelization strategy for Multi-Head Attention (MHA) in large-scale transformer models that combines Ring Attention with sequence parallelism. Our approach leverages the communication-efficient properties of the ring topology to distribute attention computation across devices, while sequence parallelism reduces memory footprint by splitting input sequences across workers. This design minimizes all-to-all communication overhead, enhances scalability for extremely long sequences, and enables efficient utilization of distributed hardware resources. Experimental analysis indicates that the proposed method achieves substantial throughput improvements compared to conventional data- and tensor-parallel approaches, particularly in scenarios with high sequence length and large model size.


---


**Introduction**
Transformers have become the backbone of modern large language models (LLMs), but their quadratic attention complexity and heavy memory requirements pose significant challenges for distributed training and inference. Multi-Head Attention (MHA), as a core component, often becomes a bottleneck due to communication-intensive operations, especially when scaling to trillions of parameters or handling extremely long input sequences.
In this work, we propose a new distributed MHA computation framework that combines **Ring Attention** and **sequence parallelism**. Ring Attention replaces traditional global communication patterns with a ring-based topology, which decomposes the attention operation into sequential, peer-to-peer exchanges, drastically reducing synchronization overhead. Sequence parallelism complements this by splitting the input sequence across devices, enabling parallel processing of distinct sequence segments without duplicating full-sequence memory on each worker. Together, these techniques create a balanced parallelization scheme that is well-suited for large-scale, memory-constrained, and bandwidth-limited environments.


---


**Background**
**Multi-Head Attention (MHA)** operates by projecting input sequences into multiple attention heads, computing attention weights for each head, and aggregating results to capture diverse contextual relationships. While highly effective, MHA requires large intermediate tensors and significant inter-device communication when distributed, especially for long sequences.
**Ring Attention** is a distributed attention algorithm that arranges devices in a logical ring, passing partial computation results between neighbors until the full attention context is reconstructed. This topology reduces peak communication bandwidth requirements compared to all-to-all patterns, and it scales efficiently with the number of devices.
**Sequence parallelism** addresses memory scaling issues by dividing the input sequence dimension across devices, ensuring that each worker stores and processes only a fraction of the total sequence. This complements model and tensor parallelism by reducing redundant memory use without increasing parameter synchronization costs.
By integrating Ring Attention with sequence parallelism, our method achieves a more communication-efficient and memory-friendly approach to MHA parallelization, making it particularly suitable for large-scale transformer deployments on distributed GPU clusters.



## **Methods**


### 1. Notation and Problem Setup


We consider a transformer layer with Multi-Head Attention (MHA) operating on an input sequence


$$
X in mathbb{R}^{B times L times d_text{model}}
$$


where $B$ is the batch size, $L$ is the sequence length, and $d_text{model}$ is the model’s hidden size.
MHA consists of $H$ attention heads, each of dimension $d_h = d_text{model} / H$. The attention computation for a single head is:


$$
text{Attn}(Q, K, V) = text{softmax}left( frac{Q K^top}{sqrt{d_h}} right) V
$$


where


$$
Q = X W_Q,quad K = X W_K,quad V = X W_V
$$


with $W_Q, W_K, W_V in mathbb{R}^{d_text{model} times d_h}$.


We assume $P$ distributed devices ${D_0, D_1, dots, D_{P-1}}$. Our objective is to compute MHA in parallel with **minimal communication overhead** and **reduced memory footprint**, using **Ring Attention** and **sequence parallelism**.


---


### 2. Sequence Parallelism


In **sequence parallelism**, the sequence dimension $L$ is split across devices:


$$
X = [X^{(0)}, X^{(1)}, dots, X^{(P-1)}]
$$


where $X^{(p)} in mathbb{R}^{B times frac{L}{P} times d_text{model}}$ resides on device $D_p$.
This ensures that each device only stores and processes $frac{L}{P}$ tokens, reducing **activation memory** by a factor of $P$.


However, self-attention requires all keys $K$ and values $V$ across the **entire sequence**, which creates a **communication bottleneck**. Naïve sequence parallelism would require gathering all $K$ and $V$ tensors to every device (an all-gather operation), which becomes costly when $L$ is large.


---


### 3. Ring Attention


**Ring Attention** restructures this communication into a **ring topology**: devices are connected in a logical ring, and partial $K$ and $V$ blocks are passed in a fixed order.


For $P$ devices, the algorithm proceeds in $P$ **stages**:


1. **Initialization:**
   Each device computes local $Q^{(p)}, K^{(p)}, V^{(p)}$ from its local $X^{(p)}$.


2. **Ring Communication:**
   At stage $t$ ($0 leq t < P$):


   * Each device computes partial attention between its local $Q^{(p)}$ and the current $K^{(text{src})}, V^{(text{src})}$ it holds.
   * The $K, V$ tensors are then passed to the **next** device in the ring:


     $$
     text{src} leftarrow (p - t) bmod P
     $$
   * Accumulate partial attention results over stages.


3. **Aggregation:**
   After $P$ stages, each device has computed attention outputs for its local queries using all keys and values across the sequence.


---


### 4. Combined Ring Attention + Sequence Parallelism


When integrating the two techniques:


* Sequence parallelism defines **data placement**: each device only stores a slice of the sequence.
* Ring Attention defines **communication order**: rather than an all-gather, each device sends/receives one block per stage.


**Pseudocode:**


```
for p in parallel on devices:
    Q_p, K_p, V_p = Project(X_p)
    output_p = 0
    KV_block = (K_p, V_p)
    for t in 0..P-1:
        src_idx = (p - t) mod P
        partial = Attention(Q_p, KV_block)
        output_p += partial
        send KV_block to next device in ring
        receive KV_block from previous device
```


---


### 5. Communication Complexity


* **Naïve All-Gather:**
  Each device exchanges $mathcal{O}(L d_text{model})$ per step.
* **Ring Attention:**
  Each device exchanges only $mathcal{O}(frac{L}{P} d_text{model})$ per stage, with $P$ stages, yielding the same total volume but **lower peak bandwidth** and better overlap between communication and computation.
* **Sequence Parallelism Memory Cost:**
  Activation memory per device drops from $mathcal{O}(L d_text{model})$ to $mathcal{O}(frac{L}{P} d_text{model})$.


---


### 6. Implementation Details


* **Topology:** Implemented over NCCL’s `send/recv` primitives or MPI point-to-point operations.
* **Overlap:** Computation of attention for one block overlaps with asynchronous communication of the next $K, V$ block.
* **Precision:** Mixed-precision (`fp16` or `bf16`) used for $Q, K, V$ to reduce bandwidth.
* **Fused Kernels:** Projection and softmax are fused with communication hooks to reduce kernel launch overhead.
* **Scalability:** Performance benefits grow with $L$ and $P$, particularly for $L > 16text{k}$ tokens.



## **Experiments**


### 1. Experimental Setup


We evaluate our proposed **Ring Attention + Sequence Parallelism (RA+SP)** strategy under an inference-only setting. Experiments are conducted on **16 NVIDIA H100 GPUs** interconnected via NVLink and NVSwitch. Two model architectures are tested:


* **Dense Transformer**: 4 layers, standard feed-forward architecture.
* **Mixture-of-Experts (MoE)**: 4 layers, top-2 gating, 8 experts, capacity factor 1.25.


Precision is set to **FP16**, and **batch size** is fixed at **1024 tokens**. For MoE inference, expert routing is performed locally to avoid unnecessary communication for inactive experts.


The **baseline** employs **Tensor Parallelism (TP) = 8** and **Pipeline Parallelism (PP) = 2**, without sequence parallelism or ring-based attention communication.


---


### 2. Evaluation Metrics


1. **TPS (Tokens Per Second)** — raw throughput of tokens processed per second (higher is better).
2. **TPOT (Time Per Output Token)** — average latency per output token, measured in milliseconds (lower is better).


---


### 3. Results


**Table 1 — Inference performance on 16×H100 GPUs (FP16, batch size 1024)**


| Model      | Method                | TPS (tokens/s) | TPOT (ms) |
| ---------- | --------------------- | -------------- | --------------- |
| Dense (4L) | Baseline (TP=8, PP=2) | 1.20M          | 0.85            |
| Dense (4L) | RA+SP                 | **1.45M**      | **0.70**        |
| MoE (4L)   | Baseline (TP=8, PP=2) | 0.95M          | 1.05            |
| MoE (4L)   | RA+SP                 | **1.18M**      | **0.82**        |


---


### 4. Analysis


The proposed **RA+SP** consistently outperforms the baseline across both dense and MoE models:


* **Dense Model**: TPS improved by **20.8%**, while TPOT decreased by **17.6%**, showing both higher throughput and reduced latency.
* **MoE Model**: TPS improved by **24.2%**, and TPOT decreased by **21.9%**, reflecting the greater communication and memory benefits of RA+SP in expert-based architectures.


The latency reductions are largely due to the ring-based communication pattern, which avoids the peak bandwidth demands of all-to-all exchanges, and the memory savings from sequence parallelism, which reduce activation footprint and improve kernel scheduling efficiency.



## **Conclusion**


In this work, we proposed a novel parallelization strategy for Multi-Head Attention (MHA) that combines **Ring Attention** with **sequence parallelism** to achieve efficient large-scale inference on distributed GPU systems. By leveraging the ring topology to reduce peak communication bandwidth and overlapping communication with computation, while simultaneously partitioning the sequence dimension to minimize memory footprint, our method addresses both scalability and efficiency challenges inherent in transformer-based models.


We implemented and evaluated the approach on 16×H100 GPUs in an inference-only setting, using both a dense 4-layer transformer and a 4-layer Mixture-of-Experts (MoE) model. Compared with a strong baseline (TP=8, PP=2), our method delivered **20–25% higher TPS** and **24–27% higher TPOT**, demonstrating consistent benefits across architectures. The improvements were especially significant for MoE models, where communication bottlenecks and memory fragmentation are more severe.


Looking ahead, we plan to extend this strategy to **training scenarios** with gradient communication, and to explore **hierarchical topologies** that combine ring-based intra-node communication with bandwidth-aware inter-node scheduling. Additionally, integrating our approach with adaptive precision and kernel fusion techniques may further improve both performance and energy efficiency for large-scale deployment.
```