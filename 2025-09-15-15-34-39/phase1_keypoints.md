# Phase 1: Keypoints Extraction - AdaSpa Paper

## Problem Statement
- Generating high-fidelity long videos with Diffusion Transformers (DiTs) is computationally expensive
- Attention mechanism is the major bottleneck with O(n²) complexity
- Example: 8-second 720p video (110K tokens) with HunyuanVideo takes ~600 PFLOPs, with ~500 PFLOPs from attention

## Key Observations about DiT Sparse Patterns

### Observation 1: Hierarchical and Blockified Structure
- DiTs exhibit hierarchical sparsity within and between different modalities (video-video, video-text, text-text)
- Attention weights matrix has clear boundaries between frames and text/video modalities
- Continuous patterns (col, diag) fail due to hierarchical discontinuities
- Blockified patterns achieve best recall (0.93-1.0 vs 0.12-0.96 for continuous patterns)

### Observation 2: Dynamic Nature
- Sparse patterns vary with inputs, layers, and heads
- Patterns remain invariant across denoising steps
- LSE distribution remains stable across denoising steps
- Offline search fails due to input-dependent variations

## Core Contributions

### 1. Comprehensive Analysis
- First systematic analysis of attention sparsity in DiTs
- Revealed two key traits: Hierarchical/Blockified structure, Step-invariant but prompt/head-adaptive

### 2. AdaSpa Method
- First Dynamic Pattern + Online Precise Search method for DiTs
- Training-free and data-free implementation
- Blockified pattern for hierarchical sparsity
- Fused LSE-Cached Search for efficient online search
- Head-adaptive Hierarchical Block Sparse Attention

### 3. Practical Implementation
- Plug-and-play solution requiring only one-line code change
- Integrates seamlessly with existing DiTs
- No additional fine-tuning or dataset-dependent profiling needed

## Key Technical Innovations

### Blockified Sparse Attention
- Partition sequence into L/B chunks with block size B
- Block-level sparse pattern MS ∈ {0,1}^(L/B × L/B)
- Optimal sparse indices by maximizing Wsum_attn within blocks

### Fused LSE-Cached Online Search
- Two-phase approach: Fused online search + LSE-Cached search
- First pass: Compute FlashAttention outputs and store LSE
- Second pass: Use cached LSE to compute Wsum_attn
- Reduces search time to <5% of full attention time

### Head-adaptive Hierarchical Strategy
- Different heads have different sparsity characteristics
- Sort heads by recall performance
- Adjust sparsity: high-recall heads get increased sparsity, low-recall heads get decreased sparsity
- Maintains average sparsity while improving accuracy

## Performance Results
- HunyuanVideo: 1.78× speedup with better quality metrics
- CogVideoX1.5-5B: 1.66× speedup with best quality metrics
- Outperforms Sparse VideoGen (1.58×, 1.52×) and MInference (1.27×, 1.39×)
- Scales to 4.01× speedup for 24-second videos
- Maintains quality across sparsity levels (0.7-0.9)