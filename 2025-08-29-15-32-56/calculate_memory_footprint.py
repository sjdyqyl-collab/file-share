#!/usr/bin/env python3

import math

# Model parameters
batch_size = 1024
num_heads = 16
head_dim = 512
hidden_size = num_heads * head_dim  # 8192
mlp_hidden_size = 32768
seq_len = 2048  # typical sequence length
vocab_size = 32000  # typical vocab size
precision = 2  # FP16 = 2 bytes

print("=== Memory Footprint Analysis ===")
print(f"Batch size: {batch_size}")
print(f"Hidden size: {hidden_size}")
print(f"MLP hidden size: {mlp_hidden_size}")
print(f"Sequence length: {seq_len}")
print()

# Calculate per-layer memory footprint
# Each transformer layer has:
# 1. Multi-head attention: 4 weight matrices (Q, K, V, O)
# 2. Feed-forward network: 2 weight matrices (up, down)
# 3. Layer norms: 2 weight matrices

# Attention weights
qkv_weight_size = 3 * hidden_size * hidden_size * precision  # Q, K, V
o_weight_size = hidden_size * hidden_size * precision  # Output projection
attention_weights = qkv_weight_size + o_weight_size

# FFN weights (dense model)
ffn_up_weight_size = hidden_size * mlp_hidden_size * precision
ffn_down_weight_size = mlp_hidden_size * hidden_size * precision
ffn_weights = ffn_up_weight_size + ffn_down_weight_size

# Layer norm weights
ln1_weight_size = hidden_size * precision  # pre-attention
ln2_weight_size = hidden_size * precision  # pre-ffn
ln_weights = ln1_weight_size + ln2_weight_size

# Total weights per layer
total_weights_per_layer = attention_weights + ffn_weights + ln_weights

# Activations per layer
# Attention: [batch, seq_len, hidden_size] * 4 (Q, K, V, attn_output)
# FFN: [batch, seq_len, mlp_hidden_size] * 2 (ffn_up, ffn_down)
attention_activations = 4 * batch_size * seq_len * hidden_size * precision
ffn_activations = 2 * batch_size * seq_len * mlp_hidden_size * precision
layer_norm_activations = 2 * batch_size * seq_len * hidden_size * precision
total_activations_per_layer = attention_activations + ffn_activations + layer_norm_activations

# Temporary buffers (conservative estimate)
buffer_size_per_layer = 0.1 * (total_weights_per_layer + total_activations_per_layer)

# Total per layer
memory_per_layer = total_weights_per_layer + total_activations_per_layer + buffer_size_per_layer

print("=== Per Layer Memory Usage ===")
print(f"Attention weights: {attention_weights / (1024**3):.2f} GB")
print(f"FFN weights: {ffn_weights / (1024**3):.2f} GB")
print(f"Layer norm weights: {ln_weights / (1024**3):.4f} GB")
print(f"Total weights per layer: {total_weights_per_layer / (1024**3):.2f} GB")
print()
print(f"Attention activations: {attention_activations / (1024**3):.2f} GB")
print(f"FFN activations: {ffn_activations / (1024**3):.2f} GB")
print(f"Total activations per layer: {total_activations_per_layer / (1024**3):.2f} GB")
print()
print(f"Total memory per layer: {memory_per_layer / (1024**3):.2f} GB")
print()

# For MoE model
experts_per_layer = 8
expert_size = (hidden_size * mlp_hidden_size * 2) * precision  # up + down
moe_ffn_weights = expert_size * experts_per_layer  # All experts
moe_gate_weights = hidden_size * experts_per_layer * precision  # Gate
moe_total_weights = attention_weights + moe_ffn_weights + moe_gate_weights + ln_weights

# MoE activations - only 1-2 experts active per token typically
active_experts = 2
moe_ffn_activations = active_experts * batch_size * seq_len * mlp_hidden_size * precision * 2
moe_total_activations = attention_activations + moe_ffn_activations + layer_norm_activations
moe_memory_per_layer = moe_total_weights + moe_total_activations + buffer_size_per_layer

print("=== MoE Model Per Layer ===")
print(f"MoE FFN weights: {moe_ffn_weights / (1024**3):.2f} GB")
print(f"Gate weights: {moe_gate_weights / (1024**3):.4f} GB")
print(f"Total MoE weights per layer: {moe_total_weights / (1024**3):.2f} GB")
print(f"Total MoE memory per layer: {moe_memory_per_layer / (1024**3):.2f} GB")
print()

# H100 SRAM/L2 cache capacity (estimated)
h100_l2_cache = 50 * 1024**3  # 50MB L2 cache
# But we need to estimate SRAM capacity (much larger)
# Based on paper context, let's assume ~2-4GB effective SRAM+L2 for model parameters
estimated_cache_capacity = 2 * 1024**3  # 2GB per GPU

print("=== Partitioning Analysis ===")
print(f"Estimated cache capacity per GPU: {estimated_cache_capacity / (1024**3):.2f} GB")
print()

# Dense model: 16 layers total
layers_per_gpu_dense = estimated_cache_capacity / memory_per_layer
print(f"Dense model: {layers_per_gpu_dense:.1f} layers per GPU")
print(f"Total GPUs needed for 16 layers: {16 / layers_per_gpu_dense:.1f}")

# MoE model: 16 layers total
layers_per_gpu_moe = estimated_cache_capacity / moe_memory_per_layer
print(f"MoE model: {layers_per_gpu_moe:.1f} layers per GPU")
print(f"Total GPUs needed for 16 layers: {16 / layers_per_gpu_moe:.1f}")

# Since we have 16 GPUs, let's determine actual partitioning
# For dense model: likely 1-2 layers per GPU
# For MoE model: likely 1 layer per GPU due to larger size

print()
print("=== Suggested Partitioning ===")
print("Dense model: 16 layers / 16 GPUs = 1 layer per GPU (but this is the naive approach)")
print("MoE model: 16 layers / 16 GPUs = 1 layer per GPU")
print()
print("However, the paper suggests GREEDY LAYER AGGREGATION:")
print("- Group contiguous layers until cache capacity is reached")
print("- With 2GB cache and ~2.6GB per layer (dense), likely 1 layer per GPU")
print("- With 2GB cache and ~21GB per layer (MoE), need further optimization")
print()
print("Revised estimate with more realistic cache usage:")
print("- Dense: 1 layer per GPU (16 partitions for 16 layers)")
print("- MoE: Need expert parallelism within layer or reduced batch size")