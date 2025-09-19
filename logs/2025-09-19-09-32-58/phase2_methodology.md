# Phase 2: Methodology Extraction

## 3.1 Draft Attention Framework

### Full Attention Definition
**Definition 3.1 (Full Attention)**: Given hidden states X ∈ ℝ^(n×d), the full attention output is:
```
Attn(X) = Softmax(QK^⊤/√d) V ∈ ℝ^(n×d)
```
where Q = XW_Q, K = XW_K, V = XW_V are projections with learned weight matrices.

### Draft Attention via Average Pooling
**Definition 3.2 (Draft Attention)**: Given hidden states X ∈ ℝ^(n×d) partitioned into g disjoint regions {R_i}:
1. **Draft Query/Key**: Average pooling over each region
   ```
   ẼQ_i = (1/|R_i|) Σ_{j∈R_i} Q_j
   ẼK_i = (1/|R_i|) Σ_{j∈R_i} K_j
   ```
2. **Draft Attention Map**: Low-resolution attention
   ```
   A_draft = Softmax(ẼQẼK^⊤/√d) ∈ ℝ^(g×g)
   ```

### Guided Sparsity Process
1. **Binary Mask Construction**: Select top-r fraction of region interactions
   - M ∈ {0,1}^(g×g) where M_ij = 1 indicates region R_i attends to R_j
   - Constructed by selecting top-scoring entries in A_draft under sparsity ratio r

2. **Token-level Mask**: Lift region sparsity to token resolution
   ```
   M̃_uv = M_ij if u ∈ R_i, v ∈ R_j
   ```

3. **Sparse Attention Computation**:
   ```
   SparseAttn(X) = Softmax((QK^⊤/√d) ⊙ M̃) V
   ```

## 3.2 Theoretical Analysis

### Error Bounds
**Theorem 3.3 (Draft Attention Error)**: For equal-sized regions |R_i| = n/g:
```
∥S - S_draft∥_F ≤ δn
```
where δ = max_{i,j} max_{u∈R_i, v∈R_j} |S_uv - ẼS_ij|

**Theorem 3.5 (Sparsity Mask Error)**: 
```
∥S - S⊙M̃∥_F ≤ n(δ + t)√(1-r)
```
where t is the threshold for top-r selection.

## 3.3 Reordering for Hardware Efficiency

### Algorithm 1: Generate Reorder Index
```
Input: Frame size (H, W), patch size (h, w), number of frames F
Output: Permutation π ∈ [n] where n = F·H·W

π ← []
for f = 0 to F-1:
    for i = 0 to H/h-1:
        for j = 0 to W/w-1:
            for u = 0 to h-1:
                for v = 0 to w-1:
                    y ← i·h + u, x ← j·w + v
                    idx ← f·H·W + y·W + x
                    Append idx to π
return π
```

### Algorithm 2: Generate Restore Index
```
Input: Permutation π ∈ [n]
Output: Inverse permutation π^(-1)
Initialize π^(-1) ← zero array of length n
for i = 0 to n-1:
    π^(-1)[π[i]] ← i
return π^(-1)
```

### Hardware Optimization Details
- **Memory Layout**: Groups tokens within h×w patches contiguously
- **Block Processing**: Enables 128 tokens per kernel to be processed as single unit
- **Efficiency**: Aligns with FlashAttention and Block Sparse Attention frameworks
- **Completeness**: Per-frame design preserves feature map integrity

## Implementation Details

### Pooling Configuration
- **Kernel**: 8×16 pooling with stride=kernel size
- **Reduction Factor**: 128× token reduction
- **Compatibility**: Matches latent sizes divisible by kernel (32×48 for 512p, 48×80 for 768p)

### Sparsity Patterns
- **Static vs Dynamic**: Dynamic per-module sparsity vs static patterns in prior work
- **Region-based**: 128-token blocks either fully computed or skipped
- **Sparsity Ratios**: Tested 55%, 60%, 75%, 80%, 90%

### Integration Strategy
- **Training-free**: No additional training required
- **Plug-and-play**: Seamless integration into existing video diffusion transformers
- **First 25% steps**: Full attention retained for quality preservation
- **Framework**: Block Sparse Attention implementation

## Computational Complexity Analysis
- **Full Attention**: O(n²d) where n = sequence length, d = hidden dimension
- **Draft Attention**: O(g²d) where g = n/128 (after 128× reduction)
- **Sparse Attention**: O(rn²d) where r = sparsity ratio (e.g., 0.1 for 90% sparsity)
- **Reordering**: O(n) linear time for permutation operations