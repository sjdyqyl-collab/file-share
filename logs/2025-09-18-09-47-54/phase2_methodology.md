# Phase 2: Methodology Extraction - DraftAttention

## 3.1 Draft Attention Framework

### Full Attention Definition
Given hidden states X ∈ ℝ^(n×d), full attention output:
```
Attn(X) = Softmax(QK^⊤/√d)V ∈ ℝ^(n×d)
```
where Q=XW_Q, K=XW_K, V=XW_V are query, key, value projections.

### Draft Attention via Average Pooling
**Input**: Hidden states X ∈ ℝ^(n×d) representing spatial-temporal tokens across frames
**Process**:
1. Partition sequence into g ≪ n disjoint regions {R_i}_{i=1}^g
2. Each region R_i corresponds to pooled spatial patch over time
3. **Draft query/key computation**:
   ```
   ˜Q_i = (1/|R_i|) Σ_{j∈R_i} Q_j
   ˜K_i = (1/|R_i|) Σ_{j∈R_i} K_j
   ```
4. **Draft attention map**:
   ```
   A_draft = Softmax(˜Q˜K^⊤/√d) ∈ ℝ^(g×g)
   ```

### Guided Sparsity via Draft Attention
1. **Mask construction**: Select top-r fraction of most salient region-to-region interactions
2. **Binary mask M ∈ {0,1}^(g×g)** where M_{ij}=1 indicates region R_i can attend to R_j
3. **Full-resolution mask ˜M ∈ {0,1}^(n×n)**:
   ```
   ˜M_{uv} = M_{ij} if u∈R_i, v∈R_j
   ```
4. **Sparse attention computation**:
   ```
   SparseAttn(X) = Softmax((QK^⊤/√d) ⊙ ˜M)V
   ```

## 3.2 Theoretical Analysis

### Error from Draft Attention
**Theorem 3.3**: For equal-sized regions |R_i| = n/g:
```
‖S - S_draft‖_F ≤ δn
```
where δ = max_{i,j} max_{u∈R_i, v∈R_j} |S_{uv} - ˜S_{ij}|

### Error from Sparsity Mask
**Theorem 3.5**: Under uniform region size:
```
‖S - S⊙˜M‖_F ≤ n(δ+t)√(1-r)
```
where t is the threshold for top-r selection.

## 3.3 Reordering for Hardware Efficiency

### Algorithm 1: Generate Reorder Index
**Input**: Frame size (H,W), patch size (h,w), number of frames F
**Output**: Permutation π ∈ [n] where n = F·H·W

**Process**:
```
π ← []
for f = 0 to F-1:
    for i = 0 to H/h-1:
        for j = 0 to W/w-1:
            for u = 0 to h-1:
                for v = 0 to w-1:
                    y ← i·h + u
                    x ← j·w + v
                    idx ← f·H·W + y·W + x
                    Append idx to π
return π
```

### Algorithm 2: Generate Restore Index
**Input**: Permutation π ∈ [n]
**Output**: Inverse permutation π^{-1}

**Process**:
```
Initialize π^{-1} ← zero array of length n
for i = 0 to n-1:
    π^{-1}[π[i]] ← i
return π^{-1}
```

### Hardware Alignment Benefits
- **Contiguous memory**: Tokens within same patch stored consecutively
- **Block-wise computation**: Enables efficient GPU kernel execution
- **Coalesced memory access**: Reduces memory bandwidth usage
- **Fixed-size blocks**: Compatible with FlashAttention and Block Sparse Attention

## Implementation Details

### Pooling Configuration
- **Kernel size**: 8×16
- **Stride**: Equal to kernel size
- **Token reduction**: 128× (8×16=128)
- **Efficiency**: Matches block size in efficient attention frameworks

### Sparsity Patterns
- **Region-level**: Entire regions (patches) either computed or skipped
- **Deterministic**: No randomness in sparsity pattern
- **Structured**: Aligns with GPU block processing

### Memory Layout
- **Before reordering**: Row-major layout (scattered tokens)
- **After reordering**: Patch-contiguous layout (grouped tokens)
- **Restoration**: Inverse permutation restores original layout

## Complexity Analysis

### Draft Attention Complexity
- **Input size**: g×g (reduced from n×n)
- **Computation**: O(g²) vs O(n²) for full attention
- **Reduction factor**: g = n/128 (for 8×16 pooling)

### Overall Complexity
- **Draft computation**: Negligible overhead (1/128² of full)
- **Sparse attention**: O(r·n²) where r is sparsity ratio
- **Reordering**: O(n) linear time permutation