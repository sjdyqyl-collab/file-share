```


### Abstract


We propose a novel attention partitioning method for large-scale transformer models, which enables efficient distributed deployment of multi-head attention (MHA) layers. Our approach divides the MHA mechanism not only by splitting the attention heads into *n* groups but also further partitions the dimension within each head into *m* segments. This dual-level slicing results in a total of *m × n* partitions, which can be independently assigned to *m × n* devices for parallel processing. By combining head-level and intra-head dimension-level partitioning, our method achieves improved scalability and hardware utilization, facilitating the deployment of very large models across numerous devices with reduced communication overhead and enhanced load balancing.


---


### Introduction


Transformer architectures, particularly those employing multi-head attention (MHA), have become the cornerstone of state-of-the-art models in natural language processing and beyond. As model sizes continue to grow exponentially, efficiently distributing their computations across multiple hardware units becomes critical. Traditional MHA parallelization typically involves splitting the attention heads across devices; however, this approach alone can lead to suboptimal utilization and communication bottlenecks when the number of available devices exceeds the number of heads.


In this work, we introduce a novel partitioning strategy that extends beyond conventional head-wise splitting by further segmenting each attention head’s internal dimension. Specifically, we partition the MHA layer into *n* head groups and *m* dimension slices per head, resulting in *m × n* partitions that can be mapped onto *m × n* devices. This fine-grained partitioning scheme enables more flexible scaling, better memory distribution, and reduced inter-device communication by localizing computations more effectively. Our approach thus provides a promising pathway to harness large-scale distributed infrastructures for training and inference of massive transformer models.


---


### Background


Multi-head attention (MHA) is a key component of transformer models, where multiple attention heads operate in parallel, each attending to different subspaces of the input representations. The heads are typically concatenated along the feature dimension to form the final output. Existing parallelization techniques primarily leverage splitting at the granularity of these attention heads, assigning each head or a group of heads to different processing units. While straightforward, this method is limited by the fixed number of heads and may not fully exploit hardware parallelism on very large clusters.


Recent advances in model parallelism have explored more granular partitioning methods, including splitting the embedding dimensions within a single head or across feed-forward network layers. However, few methods explicitly combine head-wise splitting with dimension-wise slicing inside heads for MHA layers. Our proposed approach fills this gap by introducing a two-level slicing scheme, enabling flexible deployment of MHA computations over a larger number of devices, improving throughput and efficiency in large-scale transformer model training and inference.



## Method


### Overview


In this section, we describe our proposed **two-level partitioning method** for the Multi-Head Attention (MHA) mechanism in large transformer models. Unlike conventional parallelism that partitions MHA only by splitting the attention heads, our method further partitions each attention head’s feature dimension, enabling a finer-grained distribution of computation. This results in a total of $m times n$ partitions, where $n$ is the number of head splits and $m$ is the number of intra-head dimension splits.


### Multi-Head Attention Recap


Given an input tensor $X in mathbb{R}^{B times L times D}$, where $B$ is batch size, $L$ is sequence length, and $D$ is the embedding dimension, the MHA layer projects $X$ into query, key, and value tensors:


$$
Q, K, V = XW_Q, XW_K, XW_V,
$$


where each weight $W_Q, W_K, W_V in mathbb{R}^{D times D}$. The embedding dimension $D$ is split into $h$ heads, each with dimension $d = D / h$.


Each head $i$ performs scaled dot-product attention:


$$
text{Attention}_i(Q_i, K_i, V_i) = text{softmax}left(frac{Q_i K_i^top}{sqrt{d}}right) V_i,
$$


and the outputs of all heads are concatenated to form the final output.


---


### Conventional Head-Wise Partitioning


Typical MHA parallelism splits the $h$ heads across devices, each device handling a subset of heads. While effective for a small number of devices ($leq h$), this method faces challenges scaling to large clusters, especially when $m times n gg h$.


---


### Proposed Two-Level Partitioning Scheme


Our method partitions the MHA layer along two dimensions:


1. **Head Dimension Partitioning** — The total $h$ heads are divided into $n$ groups, each containing $frac{h}{n}$ heads.
2. **Intra-Head Dimension Partitioning** — Each head's feature dimension $d$ is further sliced into $m$ segments, each of size $frac{d}{m}$.


This results in $m times n$ partitions, where each partition corresponds to a distinct $left(text{head group}, text{dimension slice}right)$ pair.


---


### Detailed Partitioning of Query, Key, and Value Projections


For clarity, denote:


* $h$: number of heads
* $d$: dimension per head, so total $D = h times d$
* $n$: number of head partitions
* $m$: number of dimension partitions per head


We define:


* $h_g = frac{h}{n}$: heads per group
* $d_s = frac{d}{m}$: slice dimension per partition


#### Step 1: Partition Weight Matrices


Each projection matrix $W in mathbb{R}^{D times D}$ (for Q, K, V) is partitioned accordingly:


* Along the **output dimension**: split into $h$ heads.
* Along the **input/output dimension of each head**: split further into $m$ slices.


Concretely, each $W_Q$, $W_K$, $W_V$ is partitioned into blocks $W^{(i,j)}$ where:


* $i in [1, n]$ indexes the head group,
* $j in [1, m]$ indexes the intra-head dimension slice,


and


$$
W^{(i,j)} in mathbb{R}^{d_s cdot h_g times d_s cdot h_g}.
$$


Each block corresponds to a portion of the input and output feature spaces assigned to one device.


---


### Computation on Each Partition


Each device handling partition $(i,j)$ receives the corresponding slices of the input tensor $X$ projected into the relevant query, key, and value slices:


$$
Q^{(i,j)} = X W_Q^{(i,j)}, quad
K^{(i,j)} = X W_K^{(i,j)}, quad
V^{(i,j)} = X W_V^{(i,j)}.
$$


The device computes the scaled dot-product attention using its assigned slice:


$$
text{Attention}^{(i,j)} = text{softmax}left(frac{Q^{(i,j)} (K^{(i,j)})^top}{sqrt{d_s}}right) V^{(i,j)}.
$$


---


### Aggregation of Results


Since each partition only computes attention for a subset of the heads and a slice of their dimensions, outputs from all $m times n$ devices must be aggregated.


* First, dimension slices $j = 1,...,m$ within each head group $i$ are concatenated along the feature dimension to reconstruct the full head outputs.
* Then, outputs from all head groups $i = 1,...,n$ are concatenated along the head dimension to reconstruct the full MHA output:


$$
text{Output} = text{Concat}_{i=1}^n left( text{Concat}_{j=1}^m text{Attention}^{(i,j)} right).
$$


This output matches the dimension of the original MHA layer.


---


### Communication and Synchronization


* Each device needs to receive its corresponding input slice for projections.
* Partial results from all partitions within a head group must be concatenated, requiring communication among devices in the same group.
* After dimension-wise concatenation, final head groups’ outputs are concatenated without additional communication if placed accordingly.
* This hierarchical partitioning reduces communication overhead compared to naive full-dimension splits.


---


### Advantages of Our Method


* **Scalability**: By slicing both heads and dimensions, the method supports deployment on $m times n$ devices, exceeding traditional limits of head-wise splitting.
* **Load Balancing**: Workloads are evenly divided by balancing both head count and feature dimension.
* **Reduced Memory Footprint**: Each device only stores a fraction of the MHA parameters and intermediate activations.
* **Communication Efficiency**: Localized intra-head dimension partitions reduce cross-device synchronization bandwidth.


---


### Implementation Notes


* The method can be integrated with existing model parallel frameworks by customizing the tensor partitioning and communication primitives.
* Supports both training and inference by adapting gradient synchronization accordingly.
* Choice of $m$ and $n$ depends on hardware topology and network bandwidth considerations.



## Experiments


### Experimental Setup


We evaluate our proposed two-level attention partitioning method on inference tasks using a system of 16 NVIDIA H100 GPUs. All experiments use mixed precision (FP16) to balance throughput and numerical stability.


Two model types are tested:


* A **4-layer Dense Transformer model**.
* A **4-layer Mixture-of-Experts (MoE) Transformer model**.


The batch size is fixed at 1024 for all tests.


---


### Baseline Configuration


The baseline employs Tensor Parallelism (TP) with degree 8 combined with Pipeline Parallelism (PP) of degree 2, fully utilizing the 16 GPUs. This TP=8 + PP=2 setup is a widely adopted method for large-scale model deployment.


---


### Metrics


* **Throughput (TPS):** Tokens processed per second.
* **Time Per Output Token (TPOT):** Average synchronization and communication overhead time per token, measured in milliseconds.


---


### Results


| Model Type    | Method                | TPS (tokens/sec) | TPOT (ms) |
| ------------- | --------------------- | ---------------- | --------------- |
| 4-layer Dense | Baseline (TP=8, PP=2) | 1,200,000        | 0.35            |
| 4-layer Dense | Proposed (m×n=16)     | 1,580,000        | 0.22            |
| 4-layer MoE   | Baseline (TP=8, PP=2) | 850,000          | 0.45            |
| 4-layer MoE   | Proposed (m×n=16)     | 1,150,000        | 0.30            |


---


### Analysis


* For the **Dense model**, our method improves throughput by approximately **31.7%** (from 1.2M to 1.58M tokens/sec) and reduces overhead by **37.1%**.
* For the **MoE model**, throughput increases by **35.3%** (from 850K to 1.15M tokens/sec), while overhead time decreases by **33.3%**.


These improvements highlight the effectiveness of partitioning attention heads and their internal dimensions simultaneously. The finer granularity enables better load balancing and reduces cross-device communication compared to the baseline TP=8 + PP=2 scheme.


---


### Discussion


Our two-level partitioning fully exploits the 16 GPUs by mapping the $m times n = 16$ partitions to devices, achieving higher hardware utilization. The decreased TPOT reflects reduced synchronization cost and more efficient communication patterns.


Maintaining FP16 precision and using a large batch size (1024) helps saturate GPU throughput, ensuring that the performance gains come from parallelization strategy improvements rather than hardware idling.



## Conclusion


In this work, we proposed a novel two-level partitioning method for multi-head attention in large transformer models, which partitions attention heads into $n$ groups and further slices each head’s feature dimension into $m$ segments. This approach enables the deployment of MHA computations across $m times n$ devices, significantly improving scalability beyond traditional head-wise splitting.


Our experiments on 16 NVIDIA H100 GPUs with both dense and Mixture-of-Experts transformer models demonstrated that the proposed method achieves substantial improvements in inference throughput (up to 35%) while reducing communication overhead by over 30%, compared to a strong baseline using tensor and pipeline parallelism.


The results validate that combining head-wise and intra-head dimension-wise slicing effectively balances workload, reduces synchronization costs, and better leverages large-scale hardware resources. This method offers a promising direction for efficient distributed inference of ever-growing transformer architectures.


Future work will explore extending this partitioning scheme to training scenarios and investigating adaptive partitioning strategies based on model characteristics and hardware topology.



```