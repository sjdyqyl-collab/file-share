import graphviz

# Create baseline DAG with TP=8, PP=2 using 16 GPUs total
dot = graphviz.Digraph('baseline_dag', comment='Baseline Transformer with TP=8, PP=2')
dot.attr(rankdir='TB', size='20,20')

# Global attributes
dot.attr('node', fontname='Arial', fontsize='10')

# Input and output nodes
dot.node('input', 'Input\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: all GPUs', 
         shape='ellipse', style='filled', fillcolor='lightblue')

dot.node('output', 'Output\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: all GPUs', 
         shape='ellipse', style='filled', fillcolor='lightblue')

# Pipeline communication between stages
dot.node('pipeline_send_0', 'Pipeline Send\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 0-7 → 8-15', 
         shape='ellipse', style='filled', fillcolor='orange')

# Layer 0 - Multi-Head Attention (GPUs 0-7)
dot.node('l0_mha_qkv', 'Layer 0 MHA QKV Projection\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nGPU: 0-7', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l0_mha_attn', 'Layer 0 MHA Attention\nInput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nOutput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nGPU: 0-7', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l0_mha_out', 'Layer 0 MHA Output Projection\nInput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 0-7', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l0_residual', 'Layer 0 Residual Add\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 0-7', 
         shape='parallelogram', style='filled', fillcolor='lightgreen')

# Layer 0 - MLP (GPUs 0-7)
dot.node('l0_mlp_fc1', 'Layer 0 MLP FC1\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, ffn_hidden=32768]\nGPU: 0-7', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l0_mlp_gelu', 'Layer 0 MLP GELU\nInput: [batch_size=1024, seq_len=10000, ffn_hidden=32768]\nOutput: [batch_size=1024, seq_len=10000, ffn_hidden=32768]\nGPU: 0-7', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l0_mlp_fc2', 'Layer 0 MLP FC2\nInput: [batch_size=1024, seq_len=10000, ffn_hidden=32768]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 0-7', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l0_mlp_residual', 'Layer 0 MLP Residual Add\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 0-7', 
         shape='parallelogram', style='filled', fillcolor='lightgreen')

# Layer 1 - Multi-Head Attention (GPUs 0-7)
dot.node('l1_mha_qkv', 'Layer 1 MHA QKV Projection\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nGPU: 0-7', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l1_mha_attn', 'Layer 1 MHA Attention\nInput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nOutput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nGPU: 0-7', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l1_mha_out', 'Layer 1 MHA Output Projection\nInput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 0-7', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l1_residual', 'Layer 1 Residual Add\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 0-7', 
         shape='parallelogram', style='filled', fillcolor='lightgreen')

# Layer 1 - MLP (GPUs 0-7)
dot.node('l1_mlp_fc1', 'Layer 1 MLP FC1\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, ffn_hidden=32768]\nGPU: 0-7', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l1_mlp_gelu', 'Layer 1 MLP GELU\nInput: [batch_size=1024, seq_len=10000, ffn_hidden=32768]\nOutput: [batch_size=1024, seq_len=10000, ffn_hidden=32768]\nGPU: 0-7', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l1_mlp_fc2', 'Layer 1 MLP FC2\nInput: [batch_size=1024, seq_len=10000, ffn_hidden=32768]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 0-7', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l1_mlp_residual', 'Layer 1 MLP Residual Add\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 0-7', 
         shape='parallelogram', style='filled', fillcolor='lightgreen')

# Layer 2 - Multi-Head Attention (GPUs 8-15)
dot.node('l2_mha_qkv', 'Layer 2 MHA QKV Projection\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nGPU: 8-15', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l2_mha_attn', 'Layer 2 MHA Attention\nInput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nOutput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nGPU: 8-15', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l2_mha_out', 'Layer 2 MHA Output Projection\nInput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 8-15', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l2_residual', 'Layer 2 Residual Add\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 8-15', 
         shape='parallelogram', style='filled', fillcolor='lightgreen')

# Layer 2 - MLP (GPUs 8-15)
dot.node('l2_mlp_fc1', 'Layer 2 MLP FC1\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, ffn_hidden=32768]\nGPU: 8-15', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l2_mlp_gelu', 'Layer 2 MLP GELU\nInput: [batch_size=1024, seq_len=10000, ffn_hidden=32768]\nOutput: [batch_size=1024, seq_len=10000, ffn_hidden=32768]\nGPU: 8-15', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l2_mlp_fc2', 'Layer 2 MLP FC2\nInput: [batch_size=1024, seq_len=10000, ffn_hidden=32768]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 8-15', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l2_mlp_residual', 'Layer 2 MLP Residual Add\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 8-15', 
         shape='parallelogram', style='filled', fillcolor='lightgreen')

# Layer 3 - Multi-Head Attention (GPUs 8-15)
dot.node('l3_mha_qkv', 'Layer 3 MHA QKV Projection\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nGPU: 8-15', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l3_mha_attn', 'Layer 3 MHA Attention\nInput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nOutput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nGPU: 8-15', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l3_mha_out', 'Layer 3 MHA Output Projection\nInput: [batch_size=1024, seq_len=10000, heads=16, d_k=512]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 8-15', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l3_residual', 'Layer 3 Residual Add\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 8-15', 
         shape='parallelogram', style='filled', fillcolor='lightgreen')

# Layer 3 - MLP (GPUs 8-15)
dot.node('l3_mlp_fc1', 'Layer 3 MLP FC1\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, ffn_hidden=32768]\nGPU: 8-15', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l3_mlp_gelu', 'Layer 3 MLP GELU\nInput: [batch_size=1024, seq_len=10000, ffn_hidden=32768]\nOutput: [batch_size=1024, seq_len=10000, ffn_hidden=32768]\nGPU: 8-15', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l3_mlp_fc2', 'Layer 3 MLP FC2\nInput: [batch_size=1024, seq_len=10000, ffn_hidden=32768]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 8-15', 
         shape='rectangle', style='filled', fillcolor='lightcoral')

dot.node('l3_mlp_residual', 'Layer 3 MLP Residual Add\nInput: [batch_size=1024, seq_len=10000, d_model=8192]\nOutput: [batch_size=1024, seq_len=10000, d_model=8192]\nGPU: 8-15', 
         shape='parallelogram', style='filled', fillcolor='lightgreen')

# Connections for baseline DAG
dot.edge('input', 'l0_mha_qkv')
dot.edge('l0_mha_qkv', 'l0_mha_attn')
dot.edge('l0_mha_attn', 'l0_mha_out')
dot.edge('input', 'l0_residual')
dot.edge('l0_mha_out', 'l0_residual')
dot.edge('l0_residual', 'l0_mlp_fc1')
dot.edge('l0_mlp_fc1', 'l0_mlp_gelu')
dot.edge('l0_mlp_gelu', 'l0_mlp_fc2')
dot.edge('l0_residual', 'l0_mlp_residual')
dot.edge('l0_mlp_fc2', 'l0_mlp_residual')
dot.edge('l0_mlp_residual', 'l1_mha_qkv')
dot.edge('l1_mha_qkv', 'l1_mha_attn')
dot.edge('l1_mha_attn', 'l1_mha_out')
dot.edge('l0_mlp_residual', 'l1_residual')
dot.edge('l1_mha_out', 'l1_residual')
dot.edge('l1_residual', 'l1_mlp_fc1')
dot.edge('l1_mlp_fc1', 'l1_mlp_gelu')
dot.edge('l1_mlp_gelu', 'l1_mlp_fc2')
dot.edge('l1_residual', 'l1_mlp_residual')
dot.edge('l1_mlp_fc2', 'l1_mlp_residual')
dot.edge('l1_mlp_residual', 'pipeline_send_0')
dot.edge('pipeline_send_0', 'l2_mha_qkv')
dot.edge('l2_mha_qkv', 'l2_mha_attn')
dot.edge('l2_mha_attn', 'l2_mha_out')
dot.edge('pipeline_send_0', 'l2_residual')
dot.edge('l2_mha_out', 'l2_residual')
dot.edge('l2_residual', 'l2_mlp_fc1')
dot.edge('l2_mlp_fc1', 'l2_mlp_gelu')
dot.edge('l2_mlp_gelu', 'l2_mlp_fc2')
dot.edge('l2_residual', 'l2_mlp_residual')
dot.edge('l2_mlp_fc2', 'l2_mlp_residual')
dot.edge('l2_mlp_residual', 'l3_mha_qkv')
dot.edge('l3_mha_qkv', 'l3_mha_attn')
dot.edge('l3_mha_attn', 'l3_mha_out')
dot.edge('l2_mlp_residual', 'l3_residual')
dot.edge('l3_mha_out', 'l3_residual')
dot.edge('l3_residual', 'l3_mlp_fc1')
dot.edge('l3_mlp_fc1', 'l3_mlp_gelu')
dot.edge('l3_mlp_gelu', 'l3_mlp_fc2')
dot.edge('l3_residual', 'l3_mlp_residual')
dot.edge('l3_mlp_fc2', 'l3_mlp_residual')
dot.edge('l3_mlp_residual', 'output')

# Save the DAG
dot.save('/home/wzc/data/file-share/2025-09-15-11-08-39/baseline_dag.dot')
dot.render('/home/wzc/data/file-share/2025-09-15-11-08-39/baseline_dag', format='svg', cleanup=False)

print("Baseline DAG saved to /home/wzc/data/file-share/2025-09-15-11-08-39/baseline_dag.dot")
print("SVG saved to /home/wzc/data/file-share/2025-09-15-11-08-39/baseline_dag.svg")