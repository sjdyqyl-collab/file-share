import graphviz

# Create baseline DAG for TP=8, PP=2 configuration
dot = graphviz.Digraph('baseline_tp8_pp2', comment='Baseline Static TP=8 PP=2 Configuration')
dot.attr(rankdir='TB', size='20,30')

# Define node styles
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')

# Input node
dot.node('input', 'Input\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: all GPUs', shape='ellipse', fillcolor='lightgreen')

# Embedding layer (split across pipeline stage 0)
dot.node('embed_0', 'Embedding Layer\\nInput: [batch_size=?, seq_len=?]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 0-7', fillcolor='lightyellow')

# Positional encoding
dot.node('pos_enc_0', 'Positional Encoding\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 0-7', fillcolor='lightyellow')

# Layer 0 - Stage 0 (GPUs 0-7)
dot.node('layer0_norm1_0', 'LayerNorm\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 0-7', fillcolor='lightcoral')
dot.node('layer0_qkv_0', 'QKV Linear (TP=8)\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, heads=32, d_k=128]\\nGPU: 0-7', fillcolor='lightcoral')
dot.node('layer0_attn_0', 'Multi-Head Attention\\nInput: [batch_size=?, seq_len=?, heads=32, d_k=128]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 0-7', fillcolor='lightcoral')
dot.node('layer0_add1_0', 'Residual Add\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 0-7', fillcolor='lightcoral')

# Layer 0 - FFN (TP=8)
dot.node('layer0_norm2_0', 'LayerNorm\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 0-7', fillcolor='lightcoral')
dot.node('layer0_ffn1_0', 'FFN Linear 1 (TP=8)\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, ffn_hidden=16384]\\nGPU: 0-7', fillcolor='lightcoral')
dot.node('layer0_gelu_0', 'GELU Activation\\nInput: [batch_size=?, seq_len=?, ffn_hidden=16384]\\nOutput: [batch_size=?, seq_len=?, ffn_hidden=16384]\\nGPU: 0-7', fillcolor='lightcoral')
dot.node('layer0_ffn2_0', 'FFN Linear 2 (TP=8)\\nInput: [batch_size=?, seq_len=?, ffn_hidden=16384]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 0-7', fillcolor='lightcoral')
dot.node('layer0_add2_0', 'Residual Add\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 0-7', fillcolor='lightcoral')

# Pipeline communication between stage 0 and 1
dot.node('pipeline_comm_0_1', 'Pipeline Communication\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 0-7 → 8-15', shape='parallelogram', fillcolor='orange')

# Layer 1 - Stage 1 (GPUs 8-15)
dot.node('layer1_norm1_1', 'LayerNorm\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 8-15', fillcolor='lightblue')
dot.node('layer1_qkv_1', 'QKV Linear (TP=8)\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, heads=32, d_k=128]\\nGPU: 8-15', fillcolor='lightblue')
dot.node('layer1_attn_1', 'Multi-Head Attention\\nInput: [batch_size=?, seq_len=?, heads=32, d_k=128]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 8-15', fillcolor='lightblue')
dot.node('layer1_add1_1', 'Residual Add\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 8-15', fillcolor='lightblue')

# Layer 1 - FFN (TP=8)
dot.node('layer1_norm2_1', 'LayerNorm\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 8-15', fillcolor='lightblue')
dot.node('layer1_ffn1_1', 'FFN Linear 1 (TP=8)\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, ffn_hidden=16384]\\nGPU: 8-15', fillcolor='lightblue')
dot.node('layer1_gelu_1', 'GELU Activation\\nInput: [batch_size=?, seq_len=?, ffn_hidden=16384]\\nOutput: [batch_size=?, seq_len=?, ffn_hidden=16384]\\nGPU: 8-15', fillcolor='lightblue')
dot.node('layer1_ffn2_1', 'FFN Linear 2 (TP=8)\\nInput: [batch_size=?, seq_len=?, ffn_hidden=16384]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 8-15', fillcolor='lightblue')
dot.node('layer1_add2_1', 'Residual Add\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 8-15', fillcolor='lightblue')

# Pipeline communication back to stage 0 for layer 2
dot.node('pipeline_comm_1_0', 'Pipeline Communication\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 8-15 → 0-7', shape='parallelogram', fillcolor='orange')

# Layer 2 - Stage 0 (GPUs 0-7)
dot.node('layer2_norm1_0', 'LayerNorm\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 0-7', fillcolor='lightcoral')
dot.node('layer2_qkv_0', 'QKV Linear (TP=8)\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, heads=32, d_k=128]\\nGPU: 0-7', fillcolor='lightcoral')
dot.node('layer2_attn_0', 'Multi-Head Attention\\nInput: [batch_size=?, seq_len=?, heads=32, d_k=128]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 0-7', fillcolor='lightcoral')
dot.node('layer2_add1_0', 'Residual Add\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 0-7', fillcolor='lightcoral')

# Layer 2 - FFN (TP=8)
dot.node('layer2_norm2_0', 'LayerNorm\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 0-7', fillcolor='lightcoral')
dot.node('layer2_ffn1_0', 'FFN Linear 1 (TP=8)\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, ffn_hidden=16384]\\nGPU: 0-7', fillcolor='lightcoral')
dot.node('layer2_gelu_0', 'GELU Activation\\nInput: [batch_size=?, seq_len=?, ffn_hidden=16384]\\nOutput: [batch_size=?, seq_len=?, ffn_hidden=16384]\\nGPU: 0-7', fillcolor='lightcoral')
dot.node('layer2_ffn2_0', 'FFN Linear 2 (TP=8)\\nInput: [batch_size=?, seq_len=?, ffn_hidden=16384]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 0-7', fillcolor='lightcoral')
dot.node('layer2_add2_0', 'Residual Add\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 0-7', fillcolor='lightcoral')

# Pipeline communication to stage 1 for layer 3
dot.node('pipeline_comm_0_1_2', 'Pipeline Communication\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 0-7 → 8-15', shape='parallelogram', fillcolor='orange')

# Layer 3 - Stage 1 (GPUs 8-15)
dot.node('layer3_norm1_1', 'LayerNorm\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 8-15', fillcolor='lightblue')
dot.node('layer3_qkv_1', 'QKV Linear (TP=8)\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, heads=32, d_k=128]\\nGPU: 8-15', fillcolor='lightblue')
dot.node('layer3_attn_1', 'Multi-Head Attention\\nInput: [batch_size=?, seq_len=?, heads=32, d_k=128]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 8-15', fillcolor='lightblue')
dot.node('layer3_add1_1', 'Residual Add\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 8-15', fillcolor='lightblue')

# Layer 3 - FFN (TP=8)
dot.node('layer3_norm2_1', 'LayerNorm\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 8-15', fillcolor='lightblue')
dot.node('layer3_ffn1_1', 'FFN Linear 1 (TP=8)\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, ffn_hidden=16384]\\nGPU: 8-15', fillcolor='lightblue')
dot.node('layer3_gelu_1', 'GELU Activation\\nInput: [batch_size=?, seq_len=?, ffn_hidden=16384]\\nOutput: [batch_size=?, seq_len=?, ffn_hidden=16384]\\nGPU: 8-15', fillcolor='lightblue')
dot.node('layer3_ffn2_1', 'FFN Linear 2 (TP=8)\\nInput: [batch_size=?, seq_len=?, ffn_hidden=16384]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 8-15', fillcolor='lightblue')
dot.node('layer3_add2_1', 'Residual Add\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 8-15', fillcolor='lightblue')

# Final LayerNorm and Output
# Communication back to stage 0 for final processing
dot.node('pipeline_comm_final', 'Pipeline Communication\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 8-15 → 0-7', shape='parallelogram', fillcolor='orange')

dot.node('final_norm', 'Final LayerNorm\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nGPU: 0-7', fillcolor='lightyellow')
dot.node('output_proj', 'Output Projection\\nInput: [batch_size=?, seq_len=?, hidden_dim=4096]\\nOutput: [batch_size=?, seq_len=?, vocab_size]\\nGPU: 0-7', fillcolor='lightyellow')

dot.node('output', 'Output\\nInput: [batch_size=?, seq_len=?, vocab_size]\\nOutput: [batch_size=?, seq_len=?, vocab_size]\\nGPU: 0-7', shape='ellipse', fillcolor='lightgreen')

# Create edges
dot.edge('input', 'embed_0')
dot.edge('embed_0', 'pos_enc_0')
dot.edge('pos_enc_0', 'layer0_norm1_0')
dot.edge('layer0_norm1_0', 'layer0_qkv_0')
dot.edge('layer0_qkv_0', 'layer0_attn_0')
dot.edge('layer0_attn_0', 'layer0_add1_0')
dot.edge('pos_enc_0', 'layer0_add1_0')
dot.edge('layer0_add1_0', 'layer0_norm2_0')
dot.edge('layer0_norm2_0', 'layer0_ffn1_0')
dot.edge('layer0_ffn1_0', 'layer0_gelu_0')
dot.edge('layer0_gelu_0', 'layer0_ffn2_0')
dot.edge('layer0_ffn2_0', 'layer0_add2_0')
dot.edge('layer0_add1_0', 'layer0_add2_0')
dot.edge('layer0_add2_0', 'pipeline_comm_0_1')

dot.edge('pipeline_comm_0_1', 'layer1_norm1_1')
dot.edge('layer1_norm1_1', 'layer1_qkv_1')
dot.edge('layer1_qkv_1', 'layer1_attn_1')
dot.edge('layer1_attn_1', 'layer1_add1_1')
dot.edge('pipeline_comm_0_1', 'layer1_add1_1')
dot.edge('layer1_add1_1', 'layer1_norm2_1')
dot.edge('layer1_norm2_1', 'layer1_ffn1_1')
dot.edge('layer1_ffn1_1', 'layer1_gelu_1')
dot.edge('layer1_gelu_1', 'layer1_ffn2_1')
dot.edge('layer1_ffn2_1', 'layer1_add2_1')
dot.edge('layer1_add1_1', 'layer1_add2_1')
dot.edge('layer1_add2_1', 'pipeline_comm_1_0')

dot.edge('pipeline_comm_1_0', 'layer2_norm1_0')
dot.edge('layer2_norm1_0', 'layer2_qkv_0')
dot.edge('layer2_qkv_0', 'layer2_attn_0')
dot.edge('layer2_attn_0', 'layer2_add1_0')
dot.edge('pipeline_comm_1_0', 'layer2_add1_0')
dot.edge('layer2_add1_0', 'layer2_norm2_0')
dot.edge('layer2_norm2_0', 'layer2_ffn1_0')
dot.edge('layer2_ffn1_0', 'layer2_gelu_0')
dot.edge('layer2_gelu_0', 'layer2_ffn2_0')
dot.edge('layer2_ffn2_0', 'layer2_add2_0')
dot.edge('layer2_add1_0', 'layer2_add2_0')
dot.edge('layer2_add2_0', 'pipeline_comm_0_1_2')

dot.edge('pipeline_comm_0_1_2', 'layer3_norm1_1')
dot.edge('layer3_norm1_1', 'layer3_qkv_1')
dot.edge('layer3_qkv_1', 'layer3_attn_1')
dot.edge('layer3_attn_1', 'layer3_add1_1')
dot.edge('pipeline_comm_0_1_2', 'layer3_add1_1')
dot.edge('layer3_add1_1', 'layer3_norm2_1')
dot.edge('layer3_norm2_1', 'layer3_ffn1_1')
dot.edge('layer3_ffn1_1', 'layer3_gelu_1')
dot.edge('layer3_gelu_1', 'layer3_ffn2_1')
dot.edge('layer3_ffn2_1', 'layer3_add2_1')
dot.edge('layer3_add1_1', 'layer3_add2_1')
dot.edge('layer3_add2_1', 'pipeline_comm_final')

dot.edge('pipeline_comm_final', 'final_norm')
dot.edge('final_norm', 'output_proj')
dot.edge('output_proj', 'output')

# Save the DAG
dot.save('/home/wzc/data/file-share/2025-09-30-10-53-27/baseline_tp8_pp2.dot')
dot.render('/home/wzc/data/file-share/2025-09-30-10-53-27/baseline_tp8_pp2', format='svg', cleanup=True)