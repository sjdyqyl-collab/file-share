# Phase 2: Methodology of DraftAttention

## Core Method Overview
DraftAttention is a two-stage attention mechanism that uses low-resolution guidance for efficient sparse attention computation in video diffusion transformers.

## Stage 1: Draft Attention Computation

### Input Processing
- Hidden states: X ∈ ℝ^(n×d) representing spatial-temporal tokens across frames
- Learned weight matrices: W_Q, W_K, W_V ∈ ℝ^(d×d)
- Projections: Q = XW_Q, K = XW_K, V = XW_V

### Down-sampling Process
1. **Partitioning**: Divide sequence into g ≪ n disjoint regions {R_i}_{i=1}^g
2. **Region Definition**: Each region R_i ⊂ [n] corresponds to pooled spatial patch over time
3. **Average Pooling**:
   - Draft query: êQ_i = (1/|R_i|) Σ_{j∈R_i} Q_j
   - Draft key: êK_i = (1/|R_i|) Σ_{j∈R_i} K_j
4. **Draft Attention Map**:
   - A_draft = Softmax(êQêK^⊤/√d) ∈ ℝ^(g×g)

### Computational Complexity
- Full attention: O(n²d)
- Draft attention: O(g²d) where g = n/128 (with 8×16 pooling)

## Stage 2: Sparse Attention with Draft Guidance

### Sparsity Pattern Generation
1. **Threshold Selection**: Select top-r fraction of region pairs based on A_draft
2. **Binary Mask Creation**: M ∈ {0,1}^(g×g) where M_ij = 1 if region pair (i,j) is selected
3. **Token-level Mask**: 
   - ĈM ∈ {0,1}^(n×n) where ĈM_uv = M_ij if u ∈ R_i, v ∈ R_j

### Reordering Algorithm
**Algorithm 1: Generate Reorder Index**
- Input: Frame size (H,W), patch size (h,w), number of frames F
- Output: Permutation π ∈ [n] where n = F·H·W
- Groups tokens within h×w patches contiguously in memory
- Ensures spatial locality for efficient block-wise computation

### Sparse Attention Computation
**Formula**:
SparseAttn(X) = Softmax((QK^⊤/√d) ⊙ ĈM) V

Where ⊙ denotes element-wise (Hadamard) product

### Restoration Process
**Algorithm 2: Generate Restore Index**
- Applies inverse permutation π⁻¹ to restore original spatial-temporal layout
- Ensures correct model inference after sparse attention computation

## Theoretical Analysis

### Error Bounds
1. **Draft Attention Error**:
   - ∥S - S_draft∥_F ≤ δn
   - Where δ = max deviation between token-level and region-averaged logits

2. **Sparsity Mask Error**:
   - ∥S - S⊙ĈM∥_F ≤ n(δ+t)√(1-r)
   - Where t is threshold for top-r selection

### Key Properties
- **Local Smoothness**: Error remains small when tokens within regions are similar
- **Concentrated Distribution**: Error small when attention mass is concentrated in few regions
- **Controlled Approximation**: Total error bounded by combination of draft and sparsity errors

## Implementation Details

### Pooling Configuration
- **Kernel**: 8×16 average pooling
- **Stride**: Equal to kernel size (non-overlapping)
- **Reduction Factor**: 128× token reduction

### Hardware Optimization
- **Block Size**: 128 tokens per kernel (matches efficient attention frameworks)
- **Memory Layout**: Contiguous storage for sparse blocks
- **GPU Kernels**: Compatible with FlashAttention and Block Sparse Attention

### Training-Free Integration
- **Plug-and-Play**: No retraining required
- **Preservation**: First 25% denoising steps use full attention
- **Compatibility**: Works with existing quantization and compression techniques