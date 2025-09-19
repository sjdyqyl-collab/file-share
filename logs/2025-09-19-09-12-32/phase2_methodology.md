# Phase 2: Methodology Extraction - DraftAttention Paper

## 3.1 Draft Attention Framework

### Full Attention Definition
Given hidden states X ∈ ℝ^(n×d), the full attention output is:
```
Attn(X) = Softmax(QK^⊤/√d) V ∈ ℝ^(n×d)
```
where Q = XW_Q, K = XW_K, V = XW_V are query, key, and value projections

### Draft Attention via Average Pooling
1. **Partitioning**: Sequence divided into g ≪ n disjoint regions {R_i}_{i=1}^g
2. **Pooling**: Draft representations computed by average pooling over each region:
   ```
   Q̃_i = (1/|R_i|) Σ_{j∈R_i} Q_j
   K̃_i = (1/|R_i|) Σ_{j∈R_i} K_j
   ```
3. **Draft Map**: Low-resolution attention map computed as:
   ```
   A_draft = Softmax(Q̃K̃^⊤/√d) ∈ ℝ^(g×g)
   ```

### Guided Sparsity Process
1. **Mask Construction**: Binary mask M ∈ {0,1}^(g×g) by selecting top-scoring entries in A_draft under sparsity ratio r
2. **Token-Level Mask**: Full-resolution mask M̃ ∈ {0,1}^(n×n) defined as:
   ```
   M̃_uv = M_ij if u ∈ R_i, v ∈ R_j
   ```
3. **Sparse Attention**: Computed as:
   ```
   SparseAttn(X) = Softmax((QK^⊤/√d) ⊙ M̃) V
   ```

## 3.2 Theoretical Analysis

### Error Bounds
1. **Draft Attention Error** (Theorem 3.3):
   ```
   ||S - S_draft||_F ≤ δn
   ```
   where δ = max_{i,j} max_{u∈R_i, v∈R_j} |S_uv - S̃_ij|

2. **Sparsity Mask Error** (Theorem 3.5):
   ```
   ||S - S⊙M̃||_F ≤ n(δ + t)√(1-r)
   ```
   where t is the threshold for top-r selection

### Key Assumptions
- Uniform region size |R_i| = n/g
- Local smoothness in videos ensures small δ
- Concentrated attention distribution ensures small t

## 3.3 Reordering Algorithm

### Algorithm 1: Generate Reorder Index
**Input**: Frame size (H, W), patch size (h, w), number of frames F
**Output**: Permutation π ∈ [n] where n = F·H·W

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
**Output**: Inverse permutation π^(-1)

```
Initialize π^(-1) ← zero array of length n
for i = 0 to n-1:
    π^(-1)[π[i]] ← i
return π^(-1)
```

### Design Rationale
- **Per-frame design**: Preserves completeness of each feature map
- **Contiguous storage**: Groups tokens within same patch together
- **Hardware alignment**: Matches efficient block size (128 tokens)
- **Memory efficiency**: Enables coalesced memory access patterns

## Implementation Details
- **Pooling kernel**: 8×16 with stride equal to kernel size
- **Token reduction**: Factor of 128 (from full resolution to draft)
- **Block size**: 128 tokens per processing unit
- **Framework**: Block Sparse Attention implementation
- **Fallback**: Full attention for first 25% of denoising steps

## Computational Complexity
- **Draft attention**: O(g²) where g = n/128
- **Sparse attention**: O(r·n²) where r is sparsity ratio
- **Reordering**: O(n) linear time permutation
- **Memory**: O(n) additional storage for indices