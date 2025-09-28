# DraftAttention: Detailed Methodology

## 3.1 Draft Attention Framework

### 3.1.1 Full Attention Definition
Given hidden states X ∈ ℝ^(n×d), the full attention output is:
```
Attn(X) = Softmax(QK^⊤/√d) V ∈ ℝ^(n×d)
```
where Q = XW_Q, K = XW_K, V = XW_V are the query, key, and value projections, and W_Q, W_K, W_V ∈ ℝ^(d×d) are learned weight matrices.

### 3.1.2 Draft Attention via Average Pooling
**Input**: Hidden states X ∈ ℝ^(n×d) representing spatial-temporal tokens across frames

**Process**:
1. Partition sequence into g ≪ n disjoint regions {R_i}_{i=1}^g
2. Each region R_i corresponds to a pooled spatial patch over time
3. Compute draft query and key via average pooling:
   ```
   Q̃_i = (1/|R_i|) Σ_{j∈R_i} Q_j
   K̃_i = (1/|R_i|) Σ_{j∈R_i} K_j
   ```
4. Compute low-resolution draft attention map:
   ```
   A_draft = Softmax(Q̃K̃^⊤/√d) ∈ ℝ^(g×g)
   ```

### 3.1.3 Guided Sparsity via Draft Attention
1. **Sparsity Pattern Extraction**:
   - Extract structured sparsity from A_draft ∈ ℝ^(g×g)
   - Retain fraction r ∈ (0,1) of most salient region-to-region interactions
   - Create binary mask M ∈ {0,1}^(g×g) where M_{ij} = 1 indicates permitted attention

2. **Token-Level Mask Construction**:
   - Lift region-level mask to token resolution:
     ```
     M̃_{uv} = M_{ij} if u ∈ R_i, v ∈ R_j
     ```

3. **Sparse Attention Computation**:
   ```
   SparseAttn(X) = Softmax((QK^⊤/√d) ⊙ M̃) V
   ```
   where ⊙ denotes element-wise product

## 3.2 Theoretical Analysis

### 3.2.1 Error from Draft Attention
**Setup**:
- Input sequence partitioned into g disjoint regions of equal size |R_i| = n/g
- Full-resolution attention logits: S_{uv} = ⟨Q_u, K_v⟩  
- Pooled approximation: S̃_{ij} = ⟨Q̃_i, K̃_j⟩ where Q̃_i = (1/|R_i|) Σ_{u∈R_i} Q_u
- Block-constant approximation: (S_draft)_{uv} = S̃_{ij} for u ∈ R_i, v ∈ R_j

**Error Bound**:
```
∥S - S_draft∥_F ≤ δn
```
where δ := max_{i,j} max_{u∈R_i, v∈R_j} |S_{uv} - S̃_{ij}|

### 3.2.2 Error from Sparsity Mask
**Setup**:
- Let S̃_(1) ≥ ... ≥ S̃_(g²) be sorted region-level scores
- Threshold: t := S̃_(⌈rg²⌉)
- Mask: M_{ij} = 1 if S̃_{ij} ≥ t, 0 otherwise
- Token-level mask: M̃_{uv} = M_{ij} for u ∈ R_i, v ∈ R_j

**Error Bound**:
```
∥S - S⊙M̃∥_F ≤ n(δ+t)√(1-r)
```

## 3.3 Reordering for Patch-Aligned Sparse Attention

### 3.3.1 Algorithm 1: Generate Reorder Index
```
Input: Frame size (H, W), patch size (h, w), number of frames F
Output: Permutation π ∈ [n] where n = F·H·W

π ← []
for f = 0 to F-1 do
    for i = 0 to H/h-1 do
        for j = 0 to W/w-1 do
            for u = 0 to h-1 do
                for v = 0 to w-1 do
                    y ← i·h + u
                    x ← j·w + v
                    idx ← f·H·W + y·W + x
                    Append idx to π
return π
```

### 3.3.2 Algorithm 2: Generate Restore Index
```
Input: Permutation π ∈ [n]
Output: Inverse permutation π^{-1}

Initialize π^{-1} ← zero array of length n
for i = 0 to n-1 do
    π^{-1}_{π_i} ← i
return π^{-1}
```

### 3.3.3 Design Details
- **Patch grouping**: Each frame divided into non-overlapping patches of size h×w
- **Memory layout**: Tokens within same patch grouped contiguously
- **Completeness**: Per-frame design preserves feature map completeness
- **Hardware alignment**: Ensures pooled regions align with memory blocks

## Computational Complexity Analysis

### Full Attention Complexity
- **Time**: O(n²d) for sequence length n and hidden dimension d
- **Memory**: O(n²) for attention matrix storage

### DraftAttention Complexity
- **Draft attention**: O(g²d) where g = n/(h×w) for patch size h×w
- **Sparse full attention**: O(rn²d) where r is sparsity ratio
- **Reordering overhead**: O(n) for permutation generation and application
- **Total**: O(g²d + rn²d) ≈ O(rn²d) since g² ≪ rn²

## Implementation Details

### Pooling Configuration
- **Kernel size**: 8×16 with stride equal to kernel size
- **Token reduction**: Factor of 128 (8×16 = 128)
- **Block size**: 128 tokens per processing unit
- **Resolution compatibility**: Optimized for 512p (32×48) and 768p (48×80) latent sizes

### Sparsity Patterns
- **Static vs dynamic**: Dynamic patterns per attention module
- **Sparsity ratios tested**: 55%, 60%, 75%, 80%, 90%
- **Quality preservation**: Full attention for first 25% of denoising steps

### Hardware Optimization
- **Framework**: Block Sparse Attention implementation
- **GPU**: H100 testing platform
- **Memory access**: Coalesced through reordering
- **Kernel launches**: Reduced through block-wise processing