# Phase 2: Methodology Extraction - DraftAttention Paper

## 3.1 Draft Attention Framework

### Full Attention Definition
Given hidden states X ∈ ℝⁿˣᵈ, full attention output:
```
Attn(X) = Softmax(QKᵀ/√d) V ∈ ℝⁿˣᵈ
```
where Q = XW_Q, K = XW_K, V = XW_V are learned projections.

### Draft Attention via Average Pooling
1. **Partition sequence** into g ≪ n disjoint regions {Rᵢ}ᵢ₌₁ᵍ
2. **Average pooling** over each region:
   ```
   Q̃ᵢ = (1/|Rᵢ|) Σⱼ∈Rᵢ Qⱼ
   K̃ᵢ = (1/|Rᵢ|) Σⱼ∈Rᵢ Kⱼ
   ```
3. **Low-resolution draft attention**:
   ```
   A_draft = Softmax(Q̃K̃ᵀ/√d) ∈ ℝᵍˣᵍ
   ```

### Guided Sparsity Construction
1. **Extract structured sparsity** by retaining fraction r ∈ (0,1) of top-scoring entries
2. **Binary mask M ∈ {0,1}ᵍˣᵍ** where Mᵢⱼ = 1 if region Rᵢ attends to Rⱼ
3. **Lift to token resolution**:
   ```
   M̃ᵤᵥ = Mᵢⱼ if u ∈ Rᵢ, v ∈ Rⱼ
   ```
4. **Sparse attention computation**:
   ```
   SparseAttn(X) = Softmax((QKᵀ/√d) ⊙ M̃) V
   ```

## 3.2 Theoretical Analysis

### Error from Draft Attention
**Theorem 3.3**: For equal-sized regions |Rᵢ| = n/g:
```
∥S - S_draft∥_F ≤ δn
```
where δ = maxᵢ,ⱼ maxᵤ∈Rᵢ,ᵥ∈Rⱼ |Sᵤᵥ - Ẽᵢⱼ|

### Error from Sparsity Mask
**Theorem 3.5**: Under uniform region size:
```
∥S - S⊙M̃∥_F ≤ n(δ + t)√(1-r)
```
where t is the threshold for top-r entries.

## 3.3 Reordering Algorithm

### Algorithm 1: Generate Reorder Index
**Input**: Frame size (H,W), patch size (h,w), number of frames F
**Output**: Permutation π ∈ [n] where n = F·H·W

```
π ← []
for f = 0 to F-1 do
    for i = 0 to H/h-1 do
        for j = 0 to W/w-1 do
            for u = 0 to h-1 do
                for v = 0 to w-1 do
                    y ← i·h + u, x ← j·w + v
                    idx ← f·H·W + y·W + x
                    Append idx to π
return π
```

### Algorithm 2: Generate Restore Index
**Input**: Permutation π ∈ [n]
**Output**: Inverse permutation π⁻¹

```
Initialize π⁻¹ ← zero array of length n
for i = 0 to n-1 do
    π⁻¹[π[i]] ← i
return π⁻¹
```

## Implementation Details

### Pooling Configuration
- **Kernel size**: 8×16 with stride equal to kernel size
- **Token reduction**: Factor of 128
- **Resolution compatibility**: 512p (32×48) and 768p (48×80) latent sizes

### Block Processing
- **128 visual tokens** within each kernel processed in single stage
- **Contiguous memory layout** for hardware efficiency
- **Fixed sparsity ratio** during inference (55%, 75%, 90% tested)

### Hardware Optimization
- **Compatible with**: FlashAttention, Block Sparse Attention
- **GPU**: H100 testing platform
- **Memory access**: Coalesced through reordering
- **Kernel launches**: Reduced through block-wise processing

## Computational Complexity

### Baseline Full Attention
- **Time complexity**: O(n²d) where n = total tokens
- **Example**: [n, d, n] matrix multiplication

### DraftAttention
- **Draft computation**: O(g²d) where g = n/128
- **Sparse attention**: O(rn²d) where r = sparsity ratio
- **Total**: O((r + 1/128)n²d)

### Communication Overhead
- **Reordering**: O(n) memory operations
- **No inter-GPU communication** in single-GPU setup