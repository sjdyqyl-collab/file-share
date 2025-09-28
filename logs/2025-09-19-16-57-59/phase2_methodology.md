# Phase 2: Detailed Methodology - DraftAttention Framework

## 1. Full Attention Definition

### Mathematical Formulation
Given hidden states X ∈ ℝ^(n×d), the full attention output is:
```
Attn(X) = Softmax(QK^⊤/√d) V ∈ ℝ^(n×d)
```
Where:
- Q = XW_Q (query projection)
- K = XW_K (key projection)  
- V = XW_V (value projection)
- W_Q, W_K, W_V ∈ ℝ^(d×d) are learned weight matrices

## 2. Draft Attention via Average Pooling

### 2.1 Token Partitioning
- **Input**: Hidden states X ∈ ℝ^(n×d) representing spatial-temporal tokens across frames
- **Partition**: Sequence divided into g ≪ n disjoint regions {R_i}_{i=1}^g
- **Region Size**: Each region R_i ⊂ [n] corresponds to pooled spatial patch over time
- **Pooling Kernel**: 8×16 with stride equal to kernel size

### 2.2 Draft Query and Key Computation
```
Q̃_i = (1/|R_i|) Σ_{j∈R_i} Q_j
K̃_i = (1/|R_i|) Σ_{j∈R_i} K_j    for i = 1,...,g
```

### 2.3 Draft Attention Map
```
A_draft = Softmax(Q̃K̃^⊤/√d) ∈ ℝ^(g×g)
```

### 2.4 Computational Complexity
- **Original**: O(n²) for full attention
- **Draft**: O(g²) where g = n/128 (128× reduction in tokens)
- **Overhead**: Minimal due to aggressive downsampling

## 3. Sparse Attention Construction

### 3.1 Mask Generation
1. **Sparsity Ratio**: Select fraction r ∈ (0,1) of most salient interactions
2. **Binary Mask**: M ∈ {0,1}^(g×g) where M_ij = 1 indicates region R_i attends to R_j
3. **Top-r Selection**: Construct mask by selecting top-scoring entries in A_draft

### 3.2 Token-level Mask Lifting
```
M̃_uv = M_ij if u ∈ R_i, v ∈ R_j
```
Where M̃ ∈ {0,1}^(n×n) is the full-resolution binary mask

### 3.3 Sparse Attention Computation
```
SparseAttn(X) = Softmax((QK^⊤/√d) ⊙ M̃) V
```
Where ⊙ denotes element-wise/Hadamard product

## 4. Deterministic Reordering Algorithm

### 4.1 Reordering Purpose
- **Memory Alignment**: Ensure contiguous storage for sparse pattern blocks
- **Hardware Efficiency**: Enable block-wise computation and coalesced memory access
- **Compatibility**: Work with FlashAttention and Block Sparse Attention

### 4.2 Algorithm 1: Generate Reorder Index
```
Input: Frame size (H, W), patch size (h, w), number of frames F
Output: Permutation π ∈ [n] where n = F·H·W

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

### 4.3 Algorithm 2: Generate Restore Index
```
Input: Permutation π ∈ [n]
Output: Inverse permutation π^(-1)

Initialize π^(-1) ← zero array of length n
for i = 0 to n-1:
    π^(-1)[π[i]] ← i
return π^(-1)
```

## 5. Theoretical Analysis

### 5.1 Error from Draft Attention
**Theorem**: For equal-sized regions |R_i| = n/g:
```
∥S - S_draft∥_F ≤ δn
```
Where:
- S_uv = ⟨Q_u, K_v⟩ (full attention logits)
- S_draft_uv = S̃_ij for u ∈ R_i, v ∈ R_j (draft approximation)
- δ = max_{i,j} max_{u∈R_i, v∈R_j} |S_uv - S̃_ij|

### 5.2 Error from Sparsity Mask
**Theorem**: Under uniform region size:
```
∥S - S⊙M̃∥_F ≤ n(δ + t)√(1-r)
```
Where:
- t = S̃_(⌈rg²⌉) (threshold for top-r selection)
- r = sparsity ratio

## 6. Implementation Details

### 6.1 Pooling Configuration
- **Kernel Size**: 8×16
- **Stride**: 8×16 (equal to kernel size)
- **Token Reduction**: 128× (8×16=128)
- **Block Size**: 128 tokens per processing unit

### 6.2 Model Architecture Integration
- **First 25% Steps**: Full attention preserved for quality
- **Remaining Steps**: DraftAttention applied
- **Framework**: Block Sparse Attention implementation
- **GPU**: H100 for latency measurements

### 6.3 Supported Resolutions
- **512p**: Latent size 32×48 (divisible by 8×16)
- **768p**: Latent size 48×80 (divisible by 8×16)
- **Padding**: Applied for non-divisible resolutions

## 7. Runtime Complexity Analysis

### 7.1 Baseline (Full Attention)
- **Computation**: [n, n, d] → O(n²d)
- **Memory**: O(n²) for attention matrix

### 7.2 Proposed (DraftAttention)
- **Draft Phase**: [g, g, d] → O(g²d) where g = n/128
- **Sparse Phase**: [n, n·r, d] → O(n²rd) where r is sparsity ratio
- **Total**: O(n²rd + g²d)

### 7.3 Speedup Factor
- **Theoretical**: ~1/r (for sparse phase)
- **Practical**: 1.75× with 90% sparsity (r=0.1)
- **Communication**: Minimal due to deterministic reordering