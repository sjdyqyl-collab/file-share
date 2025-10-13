import graphviz

# Create full attention DAG
dot = graphviz.Digraph('Full_Attention_DAG', format='svg')
dot.attr(rankdir='TB', size='20,20')

# Input layer
dot.node('input', 'Input Video Tokens', shape='ellipse', style='filled', fillcolor='lightblue')
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')

# Layer normalization
dot.node('ln1', 'LayerNorm\nInput: [batch_size, seq_len, hidden_dim]\nOutput: [batch_size, seq_len, hidden_dim]\nGPU: all GPUs')

# Multi-Head Attention - Full (no sparsity)
dot.node('q_proj', 'Query Projection\nInput: [batch_size, seq_len, hidden_dim]\nOutput: [batch_size, seq_len, hidden_dim]\nGPU: all GPUs')
dot.node('k_proj', 'Key Projection\nInput: [batch_size, seq_len, hidden_dim]\nOutput: [batch_size, seq_len, hidden_dim]\nGPU: all GPUs')
dot.node('v_proj', 'Value Projection\nInput: [batch_size, seq_len, hidden_dim]\nOutput: [batch_size, seq_len, hidden_dim]\nGPU: all GPUs')

dot.node('reshape_q', 'Reshape Q\nInput: [batch_size, seq_len, hidden_dim]\nOutput: [batch_size, seq_len, num_heads, head_dim]\nGPU: all GPUs')
dot.node('reshape_k', 'Reshape K\nInput: [batch_size, seq_len, hidden_dim]\nOutput: [batch_size, seq_len, num_heads, head_dim]\nGPU: all GPUs')
dot.node('reshape_v', 'Reshape V\nInput: [batch_size, seq_len, hidden_dim]\nOutput: [batch_size, seq_len, num_heads, head_dim]\nGPU: all GPUs')

dot.node('attn_scores', 'Attention Scores\nInput: [batch_size, seq_len, num_heads, head_dim] x2\nOutput: [batch_size, num_heads, seq_len, seq_len]\nGPU: all GPUs')

dot.node('attn_weights', 'Softmax\nInput: [batch_size, num_heads, seq_len, seq_len]\nOutput: [batch_size, num_heads, seq_len, seq_len]\nGPU: all GPUs')

dot.node('attn_output', 'Attention Output\nInput: [batch_size, num_heads, seq_len, seq_len] x [batch_size, seq_len, num_heads, head_dim]\nOutput: [batch_size, seq_len, num_heads, head_dim]\nGPU: all GPUs')

dot.node('reshape_back', 'Reshape Back\nInput: [batch_size, seq_len, num_heads, head_dim]\nOutput: [batch_size, seq_len, hidden_dim]\nGPU: all GPUs')

dot.node('out_proj', 'Output Projection\nInput: [batch_size, seq_len, hidden_dim]\nOutput: [batch_size, seq_len, hidden_dim]\nGPU: all GPUs')

# Residual connection
dot.node('residual1', 'Add & LayerNorm\nInput: [batch_size, seq_len, hidden_dim] x2\nOutput: [batch_size, seq_len, hidden_dim]\nGPU: all GPUs')

# MLP Block
dot.node('mlp_ln', 'LayerNorm\nInput: [batch_size, seq_len, hidden_dim]\nOutput: [batch_size, seq_len, hidden_dim]\nGPU: all GPUs')

dot.node('mlp_fc1', 'MLP FC1\nInput: [batch_size, seq_len, hidden_dim]\nOutput: [batch_size, seq_len, ffn_hidden_size]\nGPU: all GPUs')

dot.node('mlp_gelu', 'GELU Activation\nInput: [batch_size, seq_len, ffn_hidden_size]\nOutput: [batch_size, seq_len, ffn_hidden_size]\nGPU: all GPUs')

dot.node('mlp_fc2', 'MLP FC2\nInput: [batch_size, seq_len, ffn_hidden_size]\nOutput: [batch_size, seq_len, hidden_dim]\nGPU: all GPUs')

# Final residual connection
dot.node('residual2', 'Add\nInput: [batch_size, seq_len, hidden_dim] x2\nOutput: [batch_size, seq_len, hidden_dim]\nGPU: all GPUs')

# Output
dot.node('output', 'Output Video Tokens', shape='ellipse', style='filled', fillcolor='lightcoral')

# Connect the nodes
dot.edge('input', 'ln1')
dot.edge('ln1', 'q_proj')
dot.edge('ln1', 'k_proj')
dot.edge('ln1', 'v_proj')

dot.edge('q_proj', 'reshape_q')
dot.edge('k_proj', 'reshape_k')
dot.edge('v_proj', 'reshape_v')

dot.edge('reshape_q', 'attn_scores')
dot.edge('reshape_k', 'attn_scores')

dot.edge('attn_scores', 'attn_weights')
dot.edge('attn_weights', 'attn_output')
dot.edge('reshape_v', 'attn_output')

dot.edge('attn_output', 'reshape_back')
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
dot.render('/home/wzc/data/file-share/logs/2025-10-13-12-42-23/full_attention_dag', format='svg', cleanup=False)
dot.save('/home/wzc/data/file-share/logs/2025-10-13-12-42-23/full_attention_dag.dot')

print("Full Attention DAG saved to /home/wzc/data/file-share/logs/2025-10-13-12-42-23/full_attention_dag.svg")
print("Full Attention DOT saved to /home/wzc/data/file-share/logs/2025-10-13-12-42-23/full_attention_dag.dot")