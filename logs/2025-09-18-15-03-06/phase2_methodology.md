# DraftAttention Methodology - Technical Details

## 3.1 Draft Attention Framework

### 3.1.1 Full Attention Definition
Given hidden states X ∈ ℝ^(n×d), the full attention output is:
```
Attn(X) = Softmax(QK^⊤/√d) V ∈ ℝ^(n×d)
```
where Q = XW_Q, K = XW_K, V = XW_V are query, key, and value projections.

### 3.1.2 Draft Attention via Average Pooling
**Input**: Hidden states X ∈ ℝ^(n×d) representing spatial-temporal tokens across frames
**Process**:
1. Partition sequence into g ≪ n disjoint regions {R_i}_{i=1}^g
2. Each region R_i corresponds to pooled spatial patch over time
3. Apply average pooling to create draft representations:
   ```
   eQ_i = (1/|R_i|) Σ_{j∈R_i} Q_j
   eK_i = (1/|R_i|) Σ_{j∈R_i} K_j
   ```
4. Compute low-resolution draft attention map:
   ```
   A_draft = Softmax(eQ eK^⊤/√d) ∈ ℝ^(g×g)
   ```

### 3.1.3 Guided Sparsity Implementation
**Sparsity Pattern Extraction**:
1. Retain top fraction r ∈ (0,1) of most salient region-to-region interactions
2. Create binary mask M ∈ {0,1}^(g×g) where M_ij = 1 indicates permitted attention
3. Construct full-resolution mask cM ∈ {0,1}^(n×n):
   ```
   cM_uv = M_ij if u ∈ R_i, v ∈ R_j
   ```

**Sparse Attention Computation**:
```
SparseAttn(X) = Softmax((QK^⊤/√d) ⊙ cM) V
```

## 3.2 Theoretical Analysis

### 3.2.1 Error from Draft Attention
**Theorem 3.3**: For equal-sized regions |R_i| = n/g:
```
‖S - S_draft‖_F ≤ δn
```
where δ = max_{i,j} max_{u∈R_i, v∈R_j} |S_uv - eS_ij|

### 3.2.2 Error from Sparsity Mask
**Theorem 3.5**: Under uniform region size:
```
‖S - S⊙cM‖_F ≤ n(δ + t)√(1 - r)
```
where t is the threshold for top-r selection.

## 3.3 Reordering for Hardware Optimization

### 3.3.1 Algorithm 1: Generate Reorder Index
**Input**: Frame size (H, W), patch size (h, w), number of frames F
**Output**: Permutation π ∈ [n] where n = F·H·W

```
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

### 3.3.2 Algorithm 2: Generate Restore Index
**Input**: Permutation π ∈ [n]
**Output**: Inverse permutation π^(-1)

```
Initialize π^(-1) ← zero array of length n
for i = 0 to n-1:
    π^(-1)[π[i]] ← i
return π^(-1)
```

## 3.4 Implementation Details

### 3.4.1 Pooling Configuration
- **Kernel size**: 8×16
- **Stride**: Equal to kernel size (8×16)
- **Token reduction factor**: 128×
- **Efficient block size**: Matches FlashAttention [7] and Block Sparse Attention [18]

### 3.4.2 Sparsity Ratios Evaluated
- **Wan2.1**: 55%, 75%
- **Hunyuan**: 60%, 80%, 90%

### 3.4.3 Memory Layout Optimization
- **Per-frame design**: Preserves feature map completeness
- **Contiguous storage**: Each patch stored as contiguous block
- **Hardware alignment**: Matches GPU kernel requirements
- **Inverse restoration**: Restores original layout after computation

### 3.4.4 Computational Complexity
- **Draft attention**: O(g²) where g = n/128
- **Full attention**: O(n²·r) where r is sparsity ratio
- **Reordering overhead**: O(n) linear time
- **Total complexity**: O(n²·r + n²/128² + n)

## 3.5 Integration with Existing Models
- **Training-free**: No additional training required
- **Plug-and-play**: Direct integration with pre-trained models
- **Compatible with**: FlashAttention, Block Sparse Attention
- **Orthogonal to**: Quantization, distillation techniques