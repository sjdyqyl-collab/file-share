import graphviz

# Create baseline DiT DAG with full attention
dot = graphviz.Digraph('baseline_dit', comment='Baseline DiT with Full Attention')
dot.attr(rankdir='TB', size='20,20')

# Input processing
dot.node('input', 'Input Video Tokens', shape='ellipse', style='filled', fillcolor='lightblue')
dot.node('embed', 'Token Embedding\nInput: [batch_size=?, seq_len=?, channels=?]\nOutput: [batch_size=?, seq_len=?, hidden_size=?]', shape='rectangle', style='filled', fillcolor='lightgreen')

# Positional encoding
dot.node('pos_enc', '3D Positional Encoding\nInput: [batch_size=?, seq_len=?, hidden_size=?]\nOutput: [batch_size=?, seq_len=?, hidden_size=?]', shape='rectangle', style='filled', fillcolor='lightgreen')

# Layer 1: Multi-Head Self-Attention
dot.node('ln1_1', 'LayerNorm 1\nInput: [batch_size=?, seq_len=?, hidden_size=?]\nOutput: [batch_size=?, seq_len=?, hidden_size=?]', shape='rectangle', style='filled', fillcolor='yellow')
dot.node('q_proj1', 'Query Projection\nInput: [batch_size=?, seq_len=?, hidden_size=?]\nOutput: [batch_size=?, seq_len=?, num_heads=?, d_k=?]', shape='rectangle', style='filled', fillcolor='lightcoral')
dot.node('k_proj1', 'Key Projection\nInput: [batch_size=?, seq_len=?, hidden_size=?]\nOutput: [batch_size=?, seq_len=?, num_heads=?, d_k=?]', shape='rectangle', style='filled', fillcolor='lightcoral')
dot.node('v_proj1', 'Value Projection\nInput: [batch_size=?, seq_len=?, hidden_size=?]\nOutput: [batch_size=?, seq_len=?, num_heads=?, d_v=?]', shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('attention1', 'Full Attention\nInput Q: [batch_size=?, seq_len=?, num_heads=?, d_k=?]\nInput K: [batch_size=?, seq_len=?, num_heads=?, d_k=?]\nInput V: [batch_size=?, seq_len=?, num_heads=?, d_v=?]\nOutput: [batch_size=?, seq_len=?, num_heads=?, d_v=?]', shape='rectangle', style='filled', fillcolor='orange')

dot.node('out_proj1', 'Output Projection\nInput: [batch_size=?, seq_len=?, num_heads=?, d_v=?]\nOutput: [batch_size=?, seq_len=?, hidden_size=?]', shape='rectangle', style='filled', fillcolor='lightcoral')
dot.node('res1', 'Residual Add 1\nInput: [batch_size=?, seq_len=?, hidden_size=?] x2\nOutput: [batch_size=?, seq_len=?, hidden_size=?]', shape='parallelogram', style='filled', fillcolor='lightblue')

# Layer 1: Feed Forward Network
dot.node('ln1_2', 'LayerNorm 2\nInput: [batch_size=?, seq_len=?, hidden_size=?]\nOutput: [batch_size=?, seq_len=?, hidden_size=?]', shape='rectangle', style='filled', fillcolor='yellow')
dot.node('ffn1_1', 'FFN Linear 1\nInput: [batch_size=?, seq_len=?, hidden_size=?]\nOutput: [batch_size=?, seq_len=?, ffn_hidden=?]', shape='rectangle', style='filled', fillcolor='lightcoral')
dot.node('gelu1', 'GELU Activation\nInput: [batch_size=?, seq_len=?, ffn_hidden=?]\nOutput: [batch_size=?, seq_len=?, ffn_hidden=?]', shape='rectangle', style='filled', fillcolor='lightgreen')
dot.node('ffn1_2', 'FFN Linear 2\nInput: [batch_size=?, seq_len=?, ffn_hidden=?]\nOutput: [batch_size=?, seq_len=?, hidden_size=?]', shape='rectangle', style='filled', fillcolor='lightcoral')
dot.node('res2', 'Residual Add 2\nInput: [batch_size=?, seq_len=?, hidden_size=?] x2\nOutput: [batch_size=?, seq_len=?, hidden_size=?]', shape='parallelogram', style='filled', fillcolor='lightblue')

# Continue with more layers (simplified for baseline)
dot.node('layer_norm_final', 'Final LayerNorm\nInput: [batch_size=?, seq_len=?, hidden_size=?]\nOutput: [batch_size=?, seq_len=?, hidden_size=?]', shape='rectangle', style='filled', fillcolor='yellow')
dot.node('output_proj', 'Output Projection\nInput: [batch_size=?, seq_len=?, hidden_size=?]\nOutput: [batch_size=?, seq_len=?, output_channels=?]', shape='rectangle', style='filled', fillcolor='lightgreen')
dot.node('output', 'Output Video', shape='ellipse', style='filled', fillcolor='lightblue')

# Connect nodes
dot.edge('input', 'embed')
dot.edge('embed', 'pos_enc')
dot.edge('pos_enc', 'ln1_1')
dot.edge('ln1_1', 'q_proj1')
dot.edge('ln1_1', 'k_proj1')
dot.edge('ln1_1', 'v_proj1')
dot.edge('q_proj1', 'attention1')
dot.edge('k_proj1', 'attention1')
dot.edge('v_proj1', 'attention1')
dot.edge('attention1', 'out_proj1')
dot.edge('out_proj1', 'res1')
dot.edge('pos_enc', 'res1')  # Residual connection

dot.edge('res1', 'ln1_2')
dot.edge('ln1_2', 'ffn1_1')
dot.edge('ffn1_1', 'gelu1')
dot.edge('gelu1', 'ffn1_2')
dot.edge('ffn1_2', 'res2')
dot.edge('res1', 'res2')  # Residual connection

# Connect to final layers (simplified)
dot.edge('res2', 'layer_norm_final')
dot.edge('layer_norm_final', 'output_proj')
dot.edge('output_proj', 'output')

# Save the graph
dot.render('/home/wzc/data/file-share/2025-09-15-15-34-39/baseline_dit_dag', format='svg', cleanup=False)
dot.save('/home/wzc/data/file-share/2025-09-15-15-34-39/baseline_dit_dag.dot')

print("Baseline DiT DAG generated successfully!")