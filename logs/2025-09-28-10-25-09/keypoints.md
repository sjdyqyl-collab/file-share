# XAttention: Key Points Summary

## Abstract (Retained Original)
Long-Context Transformer Models (LCTMs) are vital for real-world applications but suffer high computational costs due to attention's quadratic complexity. Block-sparse attention mitigates this by focusing computation on critical regions, yet existing methods struggle with balancing accuracy and efficiency due to costly block importance measurements. In this paper, we introduce XAttention, a plug-and-play framework that dramatically accelerates long-context inference in Transformers models using sparse attention. XAttention's key innovation is the insight that the sum of antidiagonal values (i.e., from the lower-left to upper-right) in the attention matrix provides a powerful proxy for block importance. This allows for precise identification and pruning of non-essential blocks, resulting in high sparsity and dramatically accelerated inference. Across comprehensive evaluations on demanding long-context benchmarks—including RULER and LongBench for language, VideoMME for video understanding, and VBench for video generation—XAttention achieves accuracy comparable to full attention while delivering substantial computational gains. We demonstrate up to 13.5× acceleration in attention computation. These results underscore XAttention's ability to unlock the practical potential of block sparse attention, paving the way for scalable and efficient deployment of LCTMs in real-world applications.

## Core Problem
- Long-Context Transformer Models face quadratic computational complexity in attention mechanism
- Existing block-sparse attention methods struggle to balance accuracy vs efficiency
- Current methods have high overhead in determining block importance

## Key Innovation
- **Antidiagonal Scoring**: Using sum of antidiagonal values as proxy for block importance
- Enables precise identification and pruning of non-essential blocks
- Achieves high sparsity without accuracy loss
- Plug-and-play framework requiring no retraining

## Technical Approach
1. **Importance Prediction**: Antidiagonal sum within blocks serves as importance indicator
2. **Block Selection**: Threshold-based selection using softmax-normalized antidiagonal scores
3. **Dynamic Threshold**: Per-head threshold optimization via dynamic programming

## Performance Results
- **Speed**: Up to 13.5× acceleration in attention computation
- **Accuracy**: Maintains comparable performance to full attention
- **Sparsity**: Achieves 6-55% density depending on context length
- **Benchmarks**: Tested on RULER, LongBench, VideoMME, and VBench

## Models Evaluated
- Llama-3.1-8B-Instruct (language tasks)
- Qwen2-VL-7B-Instruct (video understanding)
- HunyuanVideo (video generation)

## Key Advantages
- Training-free implementation
- Minimal overhead in pattern selection (24.9× faster than MInference)
- Robust across different domains (text, video understanding, video generation)
- Effective for both causal and non-causal attention