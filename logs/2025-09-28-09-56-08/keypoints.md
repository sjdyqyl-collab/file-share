# XAttention: Key Points

## Abstract (Original)
Long-Context Transformer Models (LCTMs) are vital for real-world applications but suffer high computational costs due to attention's quadratic complexity. Block-sparse attention mitigates this by focusing computation on critical regions, yet existing methods struggle with balancing accuracy and efficiency due to costly block importance measurements. In this paper, we introduce XAttention, a plug-and-play framework that dramatically accelerates long-context inference in Transformers models using sparse attention. XAttention's key innovation is the insight that the sum of antidiagonal values (i.e., from the lower-left to upper-right) in the attention matrix provides a powerful proxy for block importance. This allows for precise identification and pruning of non-essential blocks, resulting in high sparsity and dramatically accelerated inference. Across comprehensive evaluations on demanding long-context benchmarks—including RULER and LongBench for language, VideoMME for video understanding, and VBench for video generation—XAttention achieves accuracy comparable to full attention while delivering substantial computational gains. We demonstrate up to 13.5× acceleration in attention computation. These results underscore XAttention's ability to unlock the practical potential of block sparse attention, paving the way for scalable and efficient deployment of LCTMs in real-world applications.

## Key Contributions

### 1. Antidiagonal Scoring Innovation
- **Core Insight**: Sum of antidiagonal values in attention matrix serves as powerful proxy for block importance
- **Advantage**: Lightweight yet effective mechanism for identifying critical attention blocks
- **Pattern Detection**: Antidiagonal pattern intersects both vertical and slash patterns within blocks

### 2. Three-Component Framework
1. **Importance Prediction**: Uses antidiagonal scoring to predict attention block importance
2. **Block Selection**: Threshold-based selection of important blocks
3. **Minimum Threshold Prediction**: Dynamic programming approach for optimal per-head thresholds

### 3. Superior Performance
- **Speed**: Up to 13.5× acceleration in attention computation
- **Accuracy**: Maintains comparable accuracy to full attention
- **Sparsity**: Achieves high sparsity (up to 93% at 128k context length)

### 4. Versatile Applications
- **Language**: RULER and LongBench benchmarks
- **Video Understanding**: VideoMME benchmark
- **Video Generation**: VBench benchmark using HunyuanVideo model

### 5. Technical Advantages
- **Training-free**: No retraining required
- **Plug-and-play**: Easy integration into existing models
- **Low overhead**: Efficient pattern selection (24.9× faster than MInference)
- **Dynamic**: Adapts to different attention heads and contexts

## Key Technical Insights

### Antidiagonal Pattern Effectiveness
- Preserves information from all tokens (each token contributes to at least one antidiagonal sum)
- Efficiently detects vertical and slash patterns crucial for attention
- Outperforms random and diagonal patterns in accuracy while maintaining lower density

### Block Selection Strategy
- Threshold-based selection outperforms Top-K and Top-Ratio approaches
- Adapts to dynamic input sequence lengths
- Maintains optimal balance between computation and accuracy

### Dynamic Threshold Prediction
- Per-head threshold optimization using dynamic programming
- Reduces average threshold from 0.9 to 0.8
- Improves both accuracy and sparsity

## Experimental Highlights

### Language Tasks (RULER)
- Outperforms full attention at several sequence lengths
- Surpasses FlexPrefill (optimal sparse baseline)
- Maintains performance up to 128k tokens

### Real-world Tasks (LongBench)
- Highest average score across all tasks
- Performance close to full attention on individual tasks

### Video Understanding (VideoMME)
- Best average performance among sparse methods
- Outperforms full attention on long video tasks

### Video Generation (VBench)
- High fidelity with PSNR up to 23.5
- Over 50% sparsity achieved
- 5-step warmup strategy preserves layout quality

## Efficiency Gains
- **Pattern Selection**: 24.9× faster than MInference, 5.9× faster than FlexPrefill
- **Attention Computation**: Up to 13.5× speedup at 256k tokens
- **Scalability**: Maintains advantages as context length increases
- **Density**: As low as 6.2% at 128k context length