# Training-free and Adaptive Sparse Attention for Efficient Long Video Generation

## Abstract
Generating high-fidelity long videos with Diffusion Transformers (DiTs) is often hindered by significant latency, primarily due to the computational demands of attention mechanisms. For instance, generating an 8-second 720p video (110K tokens) with HunyuanVideo takes about 600 PFLOPs, with around 500 PFLOPs consumed by attention computations. To address this issue, we propose AdaSpa, the first Dynamic Pattern and Online Precise Search sparse attention method. Firstly, to realize the Dynamic Pattern, we introduce a blockified pattern to efficiently capture the hierarchical sparsity inherent in DiTs. This is based on our observation that sparse characteristics of DiTs exhibit hierarchical and blockified structures between and within different modalities. This blockified approach significantly reduces the complexity of attention computation while maintaining high fidelity in the generated videos. Secondly, to enable Online Precise Search, we propose the Fused LSE-Cached Search with Head-adaptive Hierarchical Block Sparse Attention. This method is motivated by our finding that DiTs' sparse pattern and LSE vary w.r.t. inputs, layers, and heads, but remain invariant across denoising steps. By leveraging this invariance across denoising steps, it adapts to the dynamic nature of DiTs and allows for precise, real-time identification of sparse indices with minimal overhead. AdaSpa is implemented as an adaptive, plug-and-play solution and can be integrated seamlessly with existing DiTs, requiring neither additional fine-tuning nor a dataset-dependent profiling. Extensive experiments validate that AdaSpa delivers substantial acceleration across various models while preserving video quality, establishing itself as a robust and scalable approach to efficient video generation.

## 1. Introduction
Diffusion Transformers (DiTs) have set new benchmarks in video generation, enabling production of long, high-fidelity videos. However, generating high-quality videos remains computationally expensive, especially for long videos. The attention mechanism in Transformer architecture, with its O(n²) complexity, is a major bottleneck. For instance, generating an 8-second 720p video with HunyuanVideo takes about 600 PFLOPs, with nearly 500 PFLOPs consumed by attention computations.

While sparse attention mechanisms have shown success in large language models by reducing computational costs without compromising performance, existing methods face significant limitations when applied to DiTs:
- **Static Pattern** methods are not flexible enough to summarize the sparse characteristics of DiTs
- **Dynamic Pattern** methods are unable to adaptively and accurately identify the sparse patterns of DiTs

We propose AdaSpa (Adaptive Sparse Attention), the first Dynamic Pattern + Online Precise Search method for high-fidelity sparse attention. It is a training-free and data-free method designed to accelerate video generation in DiTs while preserving generation quality.

## 2. Preliminaries
### 2.1. Diffusion Transformers and 3D Full Attention
DiTs refine predictions with a diffusion process, handling multimodal data like video and text through an attention mechanism that captures spatial, temporal, and cross-modal dependencies. The total sequence length L can be represented as: L = f·h·w + t, where f is the number of latent frames, h×w is the spatial resolution of each frame, and t is the text token length.

### 2.2. FlashAttention
FlashAttention addresses memory issues by performing attention in a blockwise manner, processing smaller chunks sequentially without constructing the entire attention matrix at once.

### 2.3. Sparse Attention and Sparse Patterns
Sparse attention reduces complexity by ignoring interactions where attention weights are small. The effectiveness is evaluated using Recall, which measures how well the sparse pattern preserves the original dense attention behavior.

## 3. Sparse Pattern Characteristic in DiTs
We present key observations of sparse characteristics in DiTs:

**Observation 1**: DiTs exhibit hierarchical structure of sparse patterns within and between different modalities, making continuous patterns unsuitable. The attention weights matrix has a hierarchical organization of text and video tokens, with clear boundaries between frames.

**Observation 2**: DiTs' sparse patterns vary with inputs, layers, and heads, but remain invariant across denoising steps. This invariance enables caching strategies for efficient online search.

## 4. Methodology
### 4.1. Problem Formulation
We formulate the problem of finding optimal block sparse indices. Block Sparse Attention employs a block-wise attention method, ignoring computation of certain blocks based on sparse indices to achieve speedup.

### 4.2. Design of Adaptive Sparse Attention
AdaSpa consists of:
1. **Fused LSE-Cached Online Search**: A two-phase approach that exploits the similarity of sparse patterns across denoising steps
2. **Head-adaptive Hierarchical Block Sparse Attention**: Adapts sparsity levels for different attention heads based on their characteristics

### 4.3. Implementation
AdaSpa is implemented as a plug-and-play interface requiring only a one-line change to enable. Default configuration uses sparsity=0.8, block_size=64, Ts={10,30}.

## 5. Experiments
### 5.1. Main Results
AdaSpa consistently achieves the best performance in both quality and efficiency across all experiments:
- **HunyuanVideo**: 1.78× speedup with highest quality metrics
- **CogVideoX1.5-5B**: 1.66× speedup with best quality metrics

### 5.2. Ablation Study
- Quality remains stable across different sparsity levels
- Warmup steps significantly enhance similarity and stability
- Search strategy optimization shows optimal performance with {10,30} configuration

### 5.3. Scaling Study
AdaSpa demonstrates excellent scalability, achieving up to 4.01× speedup for 24-second videos.

## 6. Conclusion
We developed AdaSpa, a novel sparse attention approach featuring dynamic pattern and online precise search, achieving 1.78× efficiency improvement while maintaining high video quality.