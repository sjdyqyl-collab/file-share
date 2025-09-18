# Phase 1: Key Points of XAttention Paper

## Title
XAttention: Block Sparse Attention with Antidiagonal Scoring

## Core Problem
Long-Context Transformer Models (LCTMs) suffer from high computational costs due to attention's quadratic complexity, making real-world deployment challenging for long sequences.

## Key Innovation
**Antidiagonal Scoring**: The insight that the sum of antidiagonal values (lower-left to upper-right) in the attention matrix serves as a powerful proxy for block importance, enabling precise identification and pruning of non-essential blocks.

## Main Contributions
1. **Novel Scoring Method**: Antidiagonal scoring for block importance prediction that is both lightweight and accurate
2. **Plug-and-Play Framework**: Training-free approach that can be directly applied to existing models
3. **Dynamic Thresholding**: Dynamic programming approach to determine optimal thresholds per attention head
4. **Comprehensive Evaluation**: Tested across language (RULER, LongBench), video understanding (VideoMME), and video generation (VBench) tasks

## Performance Results
- **Accuracy**: Maintains accuracy comparable to full attention
- **Speed**: Achieves up to 13.5× acceleration in attention computation
- **Sparsity**: Achieves high sparsity rates (up to 93% sparse at 128k tokens)

## Technical Approach
1. **Strided Antidiagonal Scoring**: Sum values along strided antidiagonals within blocks
2. **Block Selection**: Select blocks based on threshold exceeding cumulative probability
3. **Sparse Attention**: Compute attention only on selected blocks
4. **Warmup Strategy**: Use full attention for initial denoising steps in video generation

## Models Evaluated
- **Language**: Llama-3.1-8B-Instruct
- **Video Understanding**: Qwen2-VL-7B-Instruct  
- **Video Generation**: HunyuanVideo

## Baselines Compared
- FlashAttention (dense attention)
- MInference
- FlexPrefill
- SeerAttention