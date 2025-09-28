# Phase 2: Methodology Extraction

## 1. Draft Attention Framework Overview

### 1.1 Two-Stage Attention Mechanism
- **Stage 1**: Lightweight draft attention with low-resolution representations
- **Stage 2**: Guided sparse attention on full-resolution sequence

### 1.2 Mathematical Formulations

#### Full Attention Definition
```
Attn(X) = Softmax(QK^⊤/√d)V ∈ R^(n×d)
where:
- Q = XW_Q, K = XW_K, V = XW_V
- W_Q, W_K, W_V ∈ R^(d×d) are learned weight matrices
- X ∈ R^(n×d) represents hidden states
```

#### Draft Attention via Average Pooling
```
Given hidden states X ∈ R^(n×d) partitioned into g disjoint regions {R_i}_{i=1}^g:

For each region R_i:
- Draft query:  Q̂_i = (1/|R_i|) Σ_{j∈R_i} Q_j
- Draft key: K̂_i = (1/|R_i|) Σ_{j∈R_i} K_j

Draft attention map:
A_draft = Softmax(Q̂K̂^⊤/√d) ∈ R^(g×g)
```

### 1.3 Guided Sparsity Process

#### Mask Construction
1. **Region-level mask M ∈ {0,1}^{g×g}**:
   - Select top-r fraction of entries in A_draft
   - M_ij = 1 if region R_i can attend to R_j, 0 otherwise
   - r ∈ (0,1) is sparsity ratio

2. **Token-level mask M̂ ∈ {0,1}^{n×n}**:
   - M̂_uv = M_ij if u ∈ R_i, v ∈ R_j
   - Lifts region-level sparsity to token resolution

#### Sparse Attention Computation
```
SparseAttn(X) = Softmax((QK^⊤/√d) ⊙ M̂)V
where ⊙ denotes element-wise/Hadamard product
```

## 2. Theoretical Analysis

### 2.1 Error Bounds

#### Draft Attention Error (Theorem 3.3)
```
∥S - S_draft∥_F ≤ δn
where:
- S = full attention logits matrix
- S_draft = draft attention logits matrix
- δ = max deviation between token-level and region-averaged logits
- n = sequence length
```

#### Sparsity Mask Error (Theorem 3.5)
```
∥S - S⊙M̂∥_F ≤ n(δ + t)√(1-r)
where:
- t = threshold value for top-r selection
- r = sparsity ratio
```

### 2.2 Combined Error Analysis
- Total error decomposed into:
  1. Error from average pooling (draft attention)
  2. Error from sparsity masking
- Both errors remain bounded under practical assumptions

## 3. Reordering for Hardware Efficiency

### 3.1 Problem with Default Layout
- Default row-major layout scatters spatial tokens
- Spatial patches become non-contiguous in memory
- Inefficient for sparse attention kernels

### 3.2 Deterministic Reordering Algorithm

#### Algorithm 1: Generate Reorder Index
```
Input: Frame size (H,W), patch size (h,w), frames F
Output: Permutation π ∈ [n] where n = F·H·W

π ← []
for f = 0 to F-1:
    for i = 0 to H/h-1:
        for j = 0 to W/w-1:
            for u = 0 to h-1:
                for v = 0 to w-1:
                    y = i·h + u, x = j·w + v
                    idx = f·H·W + y·W + x
                    Append idx to π
return π
```

#### Algorithm 2: Generate Restore Index
```
Input: Permutation π ∈ [n]
Output: Inverse permutation π^{-1}

Initialize π^{-1} as zero array of length n
for i = 0 to n-1:
    π^{-1}[π[i]] = i
return π^{-1}
```

### 3.3 Implementation Details
- **Patch grouping**: Tokens within h×w patches stored contiguously
- **Per-frame design**: Preserves feature map completeness
- **Memory alignment**: Enables efficient block-wise indexing and masking
- **Compatible with**: FlashAttention and Block Sparse Attention frameworks

## 4. Computational Complexity Analysis

### 4.1 Baseline Full Attention
- **Time complexity**: O(n²d)
- **Memory complexity**: O(n²)

### 4.2 Draft Attention
- **Draft computation**: O(g²d) where g = n/(h×w)
- **Sparse computation**: O(rn²d) where r is sparsity ratio
- **Total**: O(g²d + rn²d) ≪ O(n²d)

### 4.3 Practical Parameters
- **Pooling kernel**: 8×16 with stride=kernel size
- **Token reduction**: 128× (8×16=128)
- **Block size**: 128 tokens per region
- **Sparsity ratios tested**: 55%, 60%, 75%, 80%, 90%

## 5. Integration with Existing Models
- **Training-free**: No additional training required
- **Plug-and-play**: Direct integration into existing video diffusion transformers
- **Compatible with**: HunyuanVideo-T2V, Wan2.1-T2V
- **Framework support**: Block Sparse Attention implementation
- **GPU optimization**: H100 GPU tested with CUDA kernels