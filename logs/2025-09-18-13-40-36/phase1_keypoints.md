# Phase 1: Key Points Extraction - XAttention

## Original Abstract (Preserved)
Long-Context Transformer Models (LCTMs) are vital for real-world applications but suffer high computational costs due to attention's quadratic complexity. Block-sparse attention mitigates this by focusing computation on critical regions, yet existing methods struggle with balancing accuracy and efficiency due to costly block importance measurements. In this paper, we introduce XAttention, a plug-and-play framework that dramatically accelerates long-context inference in Transformers models using sparse attention. XAttention's key innovation is the insight that the sum of antidiagonal values (i.e., from the lower-left to upper-right) in the attention matrix provides a powerful proxy for block importance. This allows for precise identification and pruning of non-essential blocks, resulting in high sparsity and dramatically accelerated inference. Across comprehensive evaluations on demanding long-context benchmarks—including RULER and LongBench for language, VideoMME for video understanding, and VBench for video generation—XAttention achieves accuracy comparable to full attention while delivering substantial computational gains. We demonstrate up to 13.5× acceleration in attention computation. These results underscore XAttention's ability to unlock the practical potential of block sparse attention, paving the way for scalable and efficient deployment of LCTMs in real-world applications.

## Key Technical Contributions

### 1. Antidiagonal Scoring Method
- **Innovation**: Uses sum of antidiagonal values as proxy for block importance
- **Advantage**: Computationally efficient compared to existing pooling methods
- **Pattern Coverage**: Antidiagonal intersects both vertical and slash patterns within blocks

### 2. Block Selection Algorithm
- **Process**: Three-step approach
  1. Strided antidiagonal scoring (stride S across blocks of size B)
  2. Softmax normalization of scores
  3. Selection of minimal block set exceeding threshold τ

### 3. Dynamic Threshold Prediction
- **Method**: Dynamic programming approach for per-head threshold optimization
- **Benefit**: Adapts sparsity levels to individual attention heads
- **Implementation**: Gradual threshold reduction (10% steps) with M=1000 adjustments

## Performance Achievements
- **Speedup**: Up to 13.5× acceleration in attention computation
- **Sparsity**: Achieves 6.2-55.4% density across different context lengths
- **Accuracy**: Maintains comparable performance to full attention across benchmarks

## Evaluation Scope
- **Models**: Llama-3.1-8B-Instruct, Qwen2-VL-7B-Instruct, HunyuanVideo
- **Benchmarks**: RULER, LongBench, VideoMME, VBench
- **Context Lengths**: 4K to 256K tokens
- **Domains**: Natural language, video understanding, video generation