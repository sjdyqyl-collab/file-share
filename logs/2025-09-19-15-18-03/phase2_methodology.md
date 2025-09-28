# Phase 2: Methodology Extraction

## Core Methodology: DraftAttention Framework

### 3.1 Draft Attention Mechanism

#### Problem Formulation
- **Input**: Hidden states X ∈ ℝ^(n×d) representing spatial-temporal tokens across frames
- **Full Attention**: Attn(X) = Softmax(QK^⊤/√d)V where Q=XW_Q, K=XW_K, V=XW_V
- **Challenge**: Quadratic complexity O(n²) with respect to sequence length

#### Two-Stage Approach

**Stage 1: Low-Resolution Draft Attention**
1. **Partition**: Divide sequence into g ≪ n disjoint regions {R_i}^g_{i=1}
2. **Average Pooling**: 
   - Draft query: ȳQ_i = (1/|R_i|) Σ_{j∈R_i} Q_j
   - Draft key: ȳK_i = (1/|R_i|) Σ_{j∈R_i} K_j
3. **Draft Attention Map**: A_draft = Softmax(ȳQȳK^⊤/√d) ∈ ℝ^(g×g)

**Stage 2: Guided Sparse Attention**
1. **Sparsity Pattern Extraction**: 
   - Retain fraction r ∈ (0,1) of most salient region-to-region interactions
   - Binary mask M ∈ {0,1}^(g×g) where M_ij = 1 if region R_i attends to R_j
   - Select top-scoring entries in A_draft under fixed sparsity ratio r
2. **Token-Level Mask**: 
   - Lift region-level mask to full resolution: ĈM_uv = M_ij if u ∈ R_i, v ∈ R_j
3. **Sparse Attention Computation**: 
   - SparseAttn(X) = Softmax((QK^⊤/√d) ⊙ ĈM)V

### 3.2 Theoretical Analysis

#### Error Bounds

**Theorem 3.3 (Draft Attention Error)**
- Let δ = max_{i,j} max_{u∈R_i, v∈R_j} |S_uv - ȳS_ij|
- Frobenius-norm error bound: ||S - S_draft||_F ≤ δn
- Where S_uv = ⟨Q_u, K_v⟩ and ȳS_ij = ⟨ȳQ_i, ȳK_j⟩\n
**Theorem 3.5 (Sparsity Mask Error)**
- Let t be threshold for top-r entries in sorted region-level scores
- Error bound: ||S - S⊙ĈM||_F ≤ n(δ+t)√(1-r)
- Additional error from enforcing structured sparsity through top-r indexing

### 3.3 Reordering for Hardware Efficiency

#### Problem
- Default row-major layout scatters spatial tokens in memory
- Spatial patches become non-contiguous, hindering efficient sparse attention kernels

#### Solution: Deterministic Reordering (Algorithm 1)
1. **Patch Division**: Divide each frame into non-overlapping patches of size h×w
2. **Contiguous Grouping**: Group tokens within same patch contiguously in memory
3. **Memory Alignment**: Ensure each patch corresponds to contiguous block matching downsampled queries/keys

#### Execution Flow
1. **Forward**: Apply permutation π to ensure patch-aligned contiguous storage
2. **Attention**: Perform sparse attention on reordered tokens
3. **Restore**: Apply inverse permutation π^(-1) to restore original layout

#### Implementation Details
- **Pooling Kernel**: 8×16 with stride=kernel size (reduces tokens by 128×)
- **Block Size**: 128 visual tokens processed in single stage (computed or skipped)
- **Framework**: Uses Block Sparse Attention for GPU implementation
- **Compatibility**: Works with FlashAttention and other efficient attention frameworks

## Key Technical Innovations

1. **Dynamic Sparse Patterns**: Per-module adaptation vs. static patterns in prior work
2. **Low-Resolution Guidance**: 128× token reduction for lightweight draft computation
3. **Hardware-Aligned Reordering**: Ensures coalesced memory access and block-wise computation
4. **Training-Free**: No additional training or fine-tuning required
5. **Theoretical Guarantees**: Formal error bounds for approximation quality