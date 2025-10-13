import graphviz

# Create Compact Attention DAG
dot = graphviz.Digraph('Compact_Attention_DAG', format='svg')
dot.attr(rankdir='TB', size='25,25')

# Input layer
dot.node('input', 'Input Video Tokens\n[f, h, w, hidden_dim]', shape='ellipse', style='filled', fillcolor='lightblue')
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')

# Layer normalization
dot.node('ln1', 'LayerNorm\nInput: [batch_size, f*h*w, hidden_dim]\nOutput: [batch_size, f*h*w, hidden_dim]\nGPU: all GPUs')

# Compact Attention - Tile-based processing
dot.node('tile_group', 'Tile Grouping\nInput: [batch_size, f*h*w, hidden_dim]\nOutput: [batch_size, num_tiles, tile_size, hidden_dim]\nGPU: all GPUs', shape='parallelogram', fillcolor='yellow')

# Frame-group-wise pattern selection
dot.node('frame_group', 'Frame Group Assignment\nInput: [batch_size, num_tiles, tile_size, hidden_dim]\nOutput: Grouped by temporal distance\nGPU: all GPUs', shape='parallelogram', fillcolor='yellow')

# Multi-Head Attention with Compact Attention patterns
dot.node('q_proj', 'Query Projection\nInput: [batch_size, f*h*w, hidden_dim]\nOutput: [batch_size, f*h*w, hidden_dim]\nGPU: all GPUs')
dot.node('k_proj', 'Key Projection\nInput: [batch_size, f*h*w, hidden_dim]\nOutput: [batch_size, f*h*w, hidden_dim]\nGPU: all GPUs')
dot.node('v_proj', 'Value Projection\nInput: [batch_size, f*h*w, hidden_dim]\nOutput: [batch_size, f*h*w, hidden_dim]\nGPU: all GPUs')

dot.node('reshape_q', 'Reshape Q\nInput: [batch_size, f*h*w, hidden_dim]\nOutput: [batch_size, f*h*w, num_heads, head_dim]\nGPU: all GPUs')
dot.node('reshape_k', 'Reshape K\nInput: [batch_size, f*h*w, hidden_dim]\nOutput: [batch_size, f*h*w, num_heads, head_dim]\nGPU: all GPUs')
dot.node('reshape_v', 'Reshape V\nInput: [batch_size, f*h*w, hidden_dim]\nOutput: [batch_size, f*h*w, num_heads, head_dim]\nGPU: all GPUs')

# Compact Attention Mask Generation (offline pre-computed)
dot.node('mask_local', 'Local Pattern Mask\nPattern: Spherical around query\nCoverage: ~15% of tokens\nGPU: all GPUs', shape='hexagon', fillcolor='orange')
dot.node('mask_cross', 'Cross-shaped Pattern Mask\nPattern: Horizontal + Vertical corridors\nCoverage: ~25% of tokens\nGPU: all GPUs', shape='hexagon', fillcolor='orange')
dot.node('mask_global', 'Global Pattern Mask\nPattern: Full spatial connectivity\nCoverage: ~8% of tokens\nGPU: all GPUs', shape='hexagon', fillcolor='orange')

dot.node('mask_time_variant', 'Time-Variant Mask\nPattern: Distance-based temporal decay\nCoverage: Variable by frame distance\nGPU: all GPUs', shape='hexagon', fillcolor='orange')
dot.node('mask_time_invariant', 'Time-Invariant Mask\nPattern: Consistent across frames\nCoverage: ~30% of tokens\nGPU: all GPUs', shape='hexagon', fillcolor='orange')

# Dual Attention Windows
dot.node('dual_window', 'Dual Window Composition\nCombines local + cross patterns\nAdaptive per frame group\nGPU: all GPUs', shape='parallelogram', fillcolor='yellow')

# Compact Attention Scores (sparse computation)
dot.node('compact_scores', 'Compact Attention Scores\nInput: Q, K with masks applied\nOutput: [batch_size, num_heads, active_tokens, active_tokens]\nSparsity: 33-62%\nGPU: all GPUs', fillcolor='lightblue')

dot.node('compact_weights', 'Sparse Softmax\nInput: [batch_size, num_heads, active_tokens, active_tokens]\nOutput: [batch_size, num_heads, active_tokens, active_tokens]\nGPU: all GPUs')

dot.node('compact_output', 'Sparse Attention Output\nInput: Sparse weights x V\nOutput: [batch_size, active_tokens, num_heads, head_dim]\nGPU: all GPUs')

# Scatter back to full sequence
dot.node('scatter_back', 'Scatter to Full Sequence\nInput: [batch_size, active_tokens, num_heads, head_dim]\nOutput: [batch_size, f*h*w, num_heads, head_dim]\nGPU: all GPUs', shape='parallelogram', fillcolor='yellow')

dot.node('reshape_back', 'Reshape Back\nInput: [batch_size, f*h*w, num_heads, head_dim]\nOutput: [batch_size, f*h*w, hidden_dim]\nGPU: all GPUs')

dot.node('out_proj', 'Output Projection\nInput: [batch_size, f*h*w, hidden_dim]\nOutput: [batch_size, f*h*w, hidden_dim]\nGPU: all GPUs')

# Residual connection
dot.node('residual1', 'Add & LayerNorm\nInput: [batch_size, f*h*w, hidden_dim] x2\nOutput: [batch_size, f*h*w, hidden_dim]\nGPU: all GPUs')

# MLP Block (same as baseline)
dot.node('mlp_ln', 'LayerNorm\nInput: [batch_size, f*h*w, hidden_dim]\nOutput: [batch_size, f*h*w, hidden_dim]\nGPU: all GPUs')

dot.node('mlp_fc1', 'MLP FC1\nInput: [batch_size, f*h*w, hidden_dim]\nOutput: [batch_size, f*h*w, ffn_hidden_size]\nGPU: all GPUs')

dot.node('mlp_gelu', 'GELU Activation\nInput: [batch_size, f*h*w, ffn_hidden_size]\nOutput: [batch_size, f*h*w, ffn_hidden_size]\nGPU: all GPUs')

dot.node('mlp_fc2', 'MLP FC2\nInput: [batch_size, f*h*w, ffn_hidden_size]\nOutput: [batch_size, f*h*w, hidden_dim]\nGPU: all GPUs')

# Final residual connection
dot.node('residual2', 'Add\nInput: [batch_size, f*h*w, hidden_dim] x2\nOutput: [batch_size, f*h*w, hidden_dim]\nGPU: all GPUs')

# Output
dot.node('output', 'Output Video Tokens\n[f, h, w, hidden_dim]', shape='ellipse', style='filled', fillcolor='lightcoral')

# Connect the nodes
dot.edge('input', 'ln1')
dot.edge('ln1', 'tile_group')
dot.edge('tile_group', 'frame_group')

dot.edge('ln1', 'q_proj')
dot.edge('ln1', 'k_proj')
dot.edge('ln1', 'v_proj')

dot.edge('q_proj', 'reshape_q')
dot.edge('k_proj', 'reshape_k')
dot.edge('v_proj', 'reshape_v')

# Mask generation (dashed lines for configuration)
dot.edge('frame_group', 'mask_local', style='dashed')
dot.edge('frame_group', 'mask_cross', style='dashed')
dot.edge('frame_group', 'mask_global', style='dashed')
dot.edge('frame_group', 'mask_time_variant', style='dashed')
dot.edge('frame_group', 'mask_time_invariant', style='dashed')

dot.edge('mask_local', 'dual_window')
dot.edge('mask_cross', 'dual_window')
dot.edge('mask_time_variant', 'dual_window')
dot.edge('mask_time_invariant', 'dual_window')

dot.edge('dual_window', 'compact_scores')
dot.edge('reshape_q', 'compact_scores')
dot.edge('reshape_k', 'compact_scores')

dot.edge('compact_scores', 'compact_weights')
dot.edge('compact_weights', 'compact_output')
dot.edge('reshape_v', 'compact_output')

dot.edge('compact_output', 'scatter_back')
dot.edge('scatter_back', 'reshape_back')
dot.edge('reshape_back', 'out_proj')

dot.edge('ln1', 'residual1')
dot.edge('out_proj', 'residual1')

dot.edge('residual1', 'mlp_ln')
dot.edge('mlp_ln', 'mlp_fc1')
dot.edge('mlp_fc1', 'mlp_gelu')
dot.edge('mlp_gelu', 'mlp_fc2')

dot.edge('residual1', 'residual2')
dot.edge('mlp_fc2', 'residual2')

dot.edge('residual2', 'output')

# Save the DAG
dot.render('/home/wzc/data/file-share/logs/2025-10-13-12-42-23/compact_attention_dag', format='svg', cleanup=False)
dot.save('/home/wzc/data/file-share/logs/2025-10-13-12-42-23/compact_attention_dag.dot')

print("Compact Attention DAG saved to /home/wzc/data/file-share/logs/2025-10-13-12-42-23/compact_attention_dag.svg")
print("Compact Attention DOT saved to /home/wzc/data/file-share/logs/2025-10-13-12-42-23/compact_attention_dag.dot")