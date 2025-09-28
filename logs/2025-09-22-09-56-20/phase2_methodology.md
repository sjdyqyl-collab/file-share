# Phase 2: Methodology Extraction - DraftAttention

## Core Methodology Overview

### 1. Draft Attention Framework Architecture

#### 1.1 Two-Stage Computation Pipeline
**Stage 1: Low-Resolution Draft Attention**
- **Input**: Full-resolution queries (Q) and keys (K) from video tokens
- **Operation**: Average pooling with 8×16 kernel (stride = kernel size)
- **Reduction**: 128× token reduction (8×16 = 128 tokens pooled into 1)
- **Output**: Low-resolution draft attention map Adraft ∈ R^(g×g)

**Stage 2: Full-Resolution Sparse Attention**
- **Guidance**: Uses draft attention map to create sparsity mask
- **Sparsity Ratio**: Configurable (55%, 75%, 90% demonstrated)
- **Computation**: Block sparse attention on full-resolution tokens
- **Restoration**: Inverse reordering to restore original layout

#### 1.2 Mathematical Formulation

**Full Attention Definition**:
```
Attn(X) = Softmax(QK^T/√d)V ∈ R^(n×d)
```
Where:
- X ∈ R^(n×d): hidden states across all video frames
- Q = XW_Q, K = XW_K, V = XW_V: learned projections
- n: total number of tokens (spatial × temporal)
- d: hidden dimension

**Draft Attention via Average Pooling**:
```
For each region R_i:
    ˜Q_i = (1/|R_i|) Σ_{j∈R_i} Q_j
    ˜K_i = (1/|R_i|) Σ_{j∈R_i} K_j

A_draft = Softmax(˜Q˜K^T/√d) ∈ R^(g×g)
```

**Sparse Attention Computation**:
```
SparseAttn(X) = Softmax((QK^T/√d) ⊙ ˜M)V
```
Where ˜M is the full-resolution binary mask derived from draft attention.

### 2. Reordering Algorithm for Hardware Efficiency

#### 2.1 Token Reordering Process
**Algorithm 1: Generate Reorder Index**
- **Input**: Frame size (H,W), patch size (h,w), number of frames F
- **Process**: Groups spatial-temporal tokens into contiguous patches
- **Output**: Permutation π ∈ [n] ensuring spatial locality

**Key Steps**:
1. Divide each frame into non-overlapping h×w patches
2. Within each patch, group tokens contiguously in memory
3. Process frames sequentially maintaining temporal order
4. Ensure 128 tokens per patch align with GPU block size

#### 2.2 Memory Layout Optimization
- **Contiguous Blocks**: Each 8×16 patch becomes a contiguous memory block
- **Coalesced Access**: Enables efficient GPU memory access patterns
- **Block Processing**: 128 tokens processed as single unit (compute/skip)
- **Hardware Alignment**: Compatible with FlashAttention and Block Sparse Attention

### 3. Theoretical Analysis Framework

#### 3.1 Error Bounds for Draft Attention
**Theorem 3.3 (Draft Attention Error)**:
```
||S - S_draft||_F ≤ δn
```
Where:
- S: full-resolution attention logits
- S_draft: draft attention approximation
- δ: worst-case deviation between token and region-averaged scores
- n: total number of tokens

#### 3.2 Error Bounds for Sparsity Masking
**Theorem 3.5 (Sparsity Mask Error)**:
```
||S - S⊙˜M||_F ≤ n(δ + t)√(1-r)
```
Where:
- t: threshold for top-r sparsity selection
- r: sparsity ratio (fraction of interactions retained)
- ˜M: binary sparsity mask

### 4. Implementation Details

#### 4.1 Pooling Configuration
- **Kernel Size**: 8×16 (height × width)
- **Stride**: 8×16 (non-overlapping)
- **Reduction Factor**: 128× tokens
- **Alignment**: Compatible with latent sizes (32×48 for 512p, 48×80 for 768p)

#### 4.2 Sparsity Patterns
- **Dynamic Generation**: New pattern for each attention module
- **Top-r Selection**: Based on draft attention scores
- **Structured Sparsity**: Entire 8×16 blocks kept or skipped
- **Configurable Ratios**: 55%, 60%, 75%, 80%, 90% demonstrated

#### 4.3 Hardware Optimization
- **GPU**: H100 used for experiments
- **Attention Backend**: Block Sparse Attention framework
- **Memory Layout**: Reordered for coalesced access
- **Kernel Launches**: Minimized through block processing

### 5. Computational Complexity Analysis

#### 5.1 Baseline Complexity
- **Full Attention**: O(n²d) where n = tokens, d = hidden dimension
- **Example**: 768p video with 128 frames ≈ 614K tokens → O((614K)²d)

#### 5.2 DraftAttention Complexity
- **Draft Attention**: O(g²d) where g = n/128 → O((n/128)²d)
- **Sparse Attention**: O(rn²d) where r = sparsity ratio
- **Total**: O((n²/128² + rn²)d)

#### 5.3 Runtime Representation
- **Baseline**: [614K, d, 614K] for full attention computation
- **DraftAttention**: 
  - Draft: [4.8K, d, 4.8K] (614K/128 ≈ 4.8K)
  - Sparse: [614K, d, 61.4K] (90% sparsity → 10% of 614K)
- **Speedup**: 1.75× achieved through reduced computation and memory access