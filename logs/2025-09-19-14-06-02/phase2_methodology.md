# Phase 2: Methodology Extraction - DraftAttention Paper

## 3.1 Draft Attention Framework

### Full Attention Definition
Given hidden states X ∈ ℝ^(n×d), the full attention output is:
```
Attn(X) = Softmax(QK^⊤/√d) V ∈ ℝ^(n×d)
```
where Q = XW_Q, K = XW_K, V = XW_V are learned projections.

### Draft Attention via Average Pooling
**Input**: Hidden states X ∈ ℝ^(n×d) representing spatial-temporal tokens across frames
**Process**:
1. Partition sequence into g ≪ n disjoint regions {R_i}_{i=1}^g
2. Each region R_i corresponds to pooled spatial patch over time
3. **Draft query/key computation**:
   ```
   Q̃_i = (1/|R_i|) Σ_{j∈R_i} Q_j
   K̃_i = (1/|R_i|) Σ_{j∈R_i} K_j
   ```
4. **Low-resolution draft attention map**:
   ```
   A_draft = Softmax(Q̃K̃^⊤/√d) ∈ ℝ^(g×g)
   ```

### Guided Sparsity Construction
1. **Mask generation**: Extract top-r fraction of most salient region-to-region interactions
2. **Binary mask M ∈ {0,1}^(g×g)** where M_ij = 1 indicates permitted attention
3. **Token-level mask extension**:
   ```
   M̃_uv = M_ij if u ∈ R_i, v ∈ R_j
   ```
4. **Sparse attention computation**:
   ```
   SparseAttn(X) = Softmax((QK^⊤/√d) ⊙ M̃) V
   ```

## 3.2 Theoretical Analysis

### Error Bounds (Frobenius Norm)

#### Theorem 3.3: Draft Attention Error
**Given**: Equal-sized regions |R_i| = n/g
**Bound**: 
```
‖S - S_draft‖_F ≤ δn
```
where δ = max_{i,j} max_{u∈R_i, v∈R_j} |S_uv - S̃_ij|

#### Theorem 3.5: Sparsity Mask Error
**Bound**:
```
‖S - S⊙M̃‖_F ≤ n(δ + t)√(1-r)
```
where t = S̃_(⌈rg²⌉) is the threshold for top-r selection

## 3.3 Reordering Algorithm

### Algorithm 1: Generate Reorder Index
**Purpose**: Align memory layout with spatial region structure
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
                    append idx to π
return π
```

### Algorithm 2: Generate Restore Index
**Purpose**: Restore original spatial-temporal layout
**Input**: Permutation π ∈ [n]
**Output**: Inverse permutation π^(-1)

**Process**:
```
π^(-1) ← zero array of length n
for i = 0 to n-1:
    π^(-1)[π[i]] ← i
return π^(-1)
```

## Implementation Details

### Pooling Configuration
- **Kernel**: 8×16 average pooling
- **Stride**: Equal to kernel size (8×16)
- **Reduction factor**: 128× token reduction
- **Compatibility**: Matches efficient block size in attention frameworks

### Hardware Optimization
- **Block-wise computation**: 128 tokens processed per kernel
- **Contiguous memory layout**: Enables coalesced memory access
- **GPU kernel compatibility**: Works with FlashAttention and Block Sparse Attention

### Quality Preservation
- **Full attention retention**: First 25% of denoising steps use full attention
- **Adaptive sparsity**: Dynamic patterns per attention module
- **No training required**: Plug-and-play integration

## Computational Complexity Analysis

### Baseline Full Attention
- **Time**: O(n²d) where n = sequence length, d = hidden dimension
- **Memory**: O(n²) for attention matrix

### DraftAttention
- **Stage 1 (Draft)**: O(g²d) where g = n/128 (128× reduction)
- **Stage 2 (Sparse)**: O(rn²d) where r = sparsity ratio (e.g., 0.1 for 90% sparsity)
- **Reordering overhead**: O(n) linear time permutation

### Runtime Representation
- **Baseline**: [n, d, n] matrix multiplication
- **DraftAttention**: [g, d, g] + [n, d, rn] ≈ [n/128, d, n/128] + [n, d, 0.1n]