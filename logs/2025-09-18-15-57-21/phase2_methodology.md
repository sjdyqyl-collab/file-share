# Phase 2: Detailed Methodology

## Problem Formulation
Given hidden states X ∈ ℝ^(n×d) representing spatial-temporal tokens across frames, where n is sequence length and d is hidden dimension.

## Full Attention Baseline
**Definition 3.1 (Full Attention):**
```
Attn(X) = Softmax(QK^⊤/√d) V ∈ ℝ^(n×d)
where:
- Q = XW_Q, K = XW_K, V = XW_V
- W_Q, W_K, W_V ∈ ℝ^(d×d) are learned weight matrices
```

**Computational Complexity:** O(n²d) for matrix multiplication [n, n] × [n, d]

## Draft Attention Method

### Step 1: Draft Attention via Average Pooling
**Definition 3.2 (Draft Attention):**
```
Input: Hidden states X ∈ ℝ^(n×d)
1. Partition sequence into g ≪ n disjoint regions {R_i}_{i=1}^g
2. Each region R_i corresponds to pooled spatial patch over time
3. Compute draft query and key via average pooling:
   
   eQ_i = (1/|R_i|) Σ_{j∈R_i} Q_j
   eK_i = (1/|R_i|) Σ_{j∈R_i} K_j
   
4. Compute low-resolution draft attention map:
   A_draft = Softmax(eQ eK^⊤/√d) ∈ ℝ^(g×g)
```

**Computational Complexity:** O(g²d) where g = n/128 (with 8×16 pooling)

### Step 2: Guided Sparsity Pattern
```
1. Extract sparsity pattern from A_draft:
   - Retain fraction r ∈ (0,1) of most salient interactions
   - Create binary mask M ∈ {0,1}^(g×g)
   - M_ij = 1 if region R_i attends to R_j, 0 otherwise

2. Lift region-level mask to token resolution:
   M̃_uv = M_ij for u ∈ R_i, v ∈ R_j
   where M̃ ∈ {0,1}^(n×n)
```

### Step 3: Sparse Attention Computation
```
SparseAttn(X) = Softmax((QK^⊤/√d) ⊙ M̃) V
where ⊙ denotes element-wise/Hadamard product
```

**Computational Complexity:** O(rn²d) where r is sparsity ratio

## Theoretical Analysis

### Error Bounds

**Theorem 3.3 (Draft Attention Error):**
```
∥S - S_draft∥_F ≤ δn
where:
- S is full attention logits
- S_draft is draft attention logits
- δ = max deviation between token-level and region-averaged logits
```

**Theorem 3.5 (Sparsity Mask Error):**
```
∥S - S⊙M̃∥_F ≤ n(δ + t)√(1-r)
where:
- t = threshold for top-r selection
- r = sparsity ratio
```

## Reordering Algorithm

### Algorithm 1: Generate Reorder Index
```
Input: Frame size (H,W), patch size (h,w), frames F
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
                    append idx to π
return π
```

### Algorithm 2: Generate Restore Index
```
Input: Permutation π ∈ [n]
Output: Inverse permutation π^-1

Initialize π^-1 ← zero array of length n
for i = 0 to n-1:
    π^-1[π_i] ← i
return π^-1
```

## Implementation Details
- **Pooling Kernel:** 8×16 with stride=kernel size
- **Token Reduction:** 128× reduction (n → n/128)
- **Sparsity Ratios:** 55%, 75%, 80%, 90%
- **Hardware:** H100 GPU with Block Sparse Attention
- **Models:** HunyuanVideo-T2V (768p, 128 frames), Wan2.1-T2V (512p/768p, 80 frames)

## Runtime Comparison
- **Baseline (Full Attention):** [n, n, d] matrix multiplication
- **Proposed (DraftAttention):** 
  - Draft attention: [n/128, n/128, d]
  - Sparse attention: [n, n, d] with sparsity ratio r
- **Total Runtime:** O(rn²d) + O((n/128)²d)