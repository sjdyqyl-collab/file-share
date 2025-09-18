import graphviz

# Create baseline DAG with Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
dot = graphviz.Digraph('baseline_dense_transformer', comment='Dense Transformer Baseline with TP=8, PP=2')
dot.attr(rankdir='TB', size='20,20')

# Define node styles
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
dot.attr('edge', fontsize='10')

# Global input
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='Global Input', style='dashed')
    c.node('input_tokens', 'Input Tokens\nInput: [batch_size=1024, seq_len=10000]\nGPU: all GPUs', shape='ellipse', fillcolor='lightgreen')
    c.node('embedding', 'Token Embedding\nInput: [batch_size=1024, seq_len=10000]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: all GPUs', shape='rectangle')
    c.node('position_embed', 'Position Embedding\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: all GPUs', shape='rectangle')

# Pipeline Stage 0 (Devices 0-7)
with dot.subgraph(name='cluster_stage0') as c:
    c.attr(label='Pipeline Stage 0 (Devices 0-7)', style='rounded')
    
    # Layer 0
    with c.subgraph(name='cluster_layer0') as layer:
        layer.attr(label='Layer 0')
        
        # LayerNorm 0
        layer.node('ln0', 'LayerNorm 0\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: 0-7', fillcolor='lightyellow')
        
        # QKV projections for each device
        for i in range(8):
            layer.node(f'qkv0_{i}', f'QKV Projection {i}\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, qkv=1024]\nGPU: {i}', fillcolor='lightcoral')
        
        # Attention computation for each device
        for i in range(8):
            layer.node(f'attn0_{i}', f'Scaled Dot-Product Attention {i}\nInput: [batch_size=1024, seq_len=10000, heads=2, head_dim=512]\nOutput: [batch_size=1024, seq_len=10000, heads=2, head_dim=512]\nGPU: {i}', fillcolor='lightcoral')
        
        # Output projections for each device
        for i in range(8):
            layer.node(f'out0_{i}', f'Output Projection {i}\nInput: [batch_size=1024, seq_len=10000, hidden=1024]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: {i}', fillcolor='lightcoral')
        
        # All-reduce for attention output
        layer.node('ar0', 'All-Reduce\nInput: [batch_size=1024, seq_len=10000, hidden=8192]×8\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: 0-7', shape='parallelogram', fillcolor='orange')
        
        # Residual connection
        layer.node('res0', 'Residual Add\nInput: [batch_size=1024, seq_len=10000, hidden=8192]×2\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: 0-7', shape='ellipse', fillcolor='lightgreen')
        
        # LayerNorm 1
        layer.node('ln1', 'LayerNorm 1\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: 0-7', fillcolor='lightyellow')
        
        # MLP components
        for i in range(8):
            layer.node(f'mlp1_0_{i}', f'MLP Linear 4h {i}\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, ffn=4096]\nGPU: {i}', fillcolor='lightblue')
            layer.node(f'gelu0_{i}', f'GELU {i}\nInput: [batch_size=1024, seq_len=10000, ffn=4096]\nOutput: [batch_size=1024, seq_len=10000, ffn=4096]\nGPU: {i}', fillcolor='lightblue')
            layer.node(f'mlp2_0_{i}', f'MLP Linear h {i}\nInput: [batch_size=1024, seq_len=10000, ffn=4096]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: {i}', fillcolor='lightblue')
        
        # All-reduce for MLP output
        layer.node('ar1', 'All-Reduce\nInput: [batch_size=1024, seq_len=10000, hidden=8192]×8\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: 0-7', shape='parallelogram', fillcolor='orange')
        
        # Residual connection
        layer.node('res1', 'Residual Add\nInput: [batch_size=1024, seq_len=10000, hidden=8192]×2\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: 0-7', shape='ellipse', fillcolor='lightgreen')

    # Layer 1
    with c.subgraph(name='cluster_layer1') as layer:
        layer.attr(label='Layer 1')
        
        layer.node('ln2', 'LayerNorm 2\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: 0-7', fillcolor='lightyellow')
        
        for i in range(8):
            layer.node(f'qkv1_{i}', f'QKV Projection {i}\nGPU: {i}', fillcolor='lightcoral')
            layer.node(f'attn1_{i}', f'Attention {i}\nGPU: {i}', fillcolor='lightcoral')
            layer.node(f'out1_{i}', f'Output Projection {i}\nGPU: {i}', fillcolor='lightcoral')
        
        layer.node('ar2', 'All-Reduce\nGPU: 0-7', shape='parallelogram', fillcolor='orange')
        layer.node('res2', 'Residual Add\nGPU: 0-7', shape='ellipse', fillcolor='lightgreen')
        
        layer.node('ln3', 'LayerNorm 3\nGPU: 0-7', fillcolor='lightyellow')
        
        for i in range(8):
            layer.node(f'mlp1_1_{i}', f'MLP Linear 4h {i}\nGPU: {i}', fillcolor='lightblue')
            layer.node(f'gelu1_{i}', f'GELU {i}\nGPU: {i}', fillcolor='lightblue')
            layer.node(f'mlp2_1_{i}', f'MLP Linear h {i}\nGPU: {i}', fillcolor='lightblue')
        
        layer.node('ar3', 'All-Reduce\nGPU: 0-7', shape='parallelogram', fillcolor='orange')
        layer.node('res3', 'Residual Add\nGPU: 0-7', shape='ellipse', fillcolor='lightgreen')

# Pipeline communication
with dot.subgraph(name='cluster_pipeline_comm') as c:
    c.attr(label='Pipeline Communication', style='dashed')
    c.node('pipe_send', 'Send Activations\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: 0-7 → 8-15', shape='parallelogram', fillcolor='orange')

# Pipeline Stage 1 (Devices 8-15)
with dot.subgraph(name='cluster_stage1') as c:
    c.attr(label='Pipeline Stage 1 (Devices 8-15)', style='rounded')
    
    # Layer 2
    with c.subgraph(name='cluster_layer2') as layer:
        layer.attr(label='Layer 2')
        
        layer.node('ln4', 'LayerNorm 4\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: 8-15', fillcolor='lightyellow')
        
        for i in range(8, 16):
            layer.node(f'qkv2_{i}', f'QKV Projection {i}\nGPU: {i}', fillcolor='lightcoral')
            layer.node(f'attn2_{i}', f'Attention {i}\nGPU: {i}', fillcolor='lightcoral')
            layer.node(f'out2_{i}', f'Output Projection {i}\nGPU: {i}', fillcolor='lightcoral')
        
        layer.node('ar4', 'All-Reduce\nGPU: 8-15', shape='parallelogram', fillcolor='orange')
        layer.node('res4', 'Residual Add\nGPU: 8-15', shape='ellipse', fillcolor='lightgreen')
        
        layer.node('ln5', 'LayerNorm 5\nGPU: 8-15', fillcolor='lightyellow')
        
        for i in range(8, 16):
            layer.node(f'mlp1_2_{i}', f'MLP Linear 4h {i}\nGPU: {i}', fillcolor='lightblue')
            layer.node(f'gelu2_{i}', f'GELU {i}\nGPU: {i}', fillcolor='lightblue')
            layer.node(f'mlp2_2_{i}', f'MLP Linear h {i}\nGPU: {i}', fillcolor='lightblue')
        
        layer.node('ar5', 'All-Reduce\nGPU: 8-15', shape='parallelogram', fillcolor='orange')
        layer.node('res5', 'Residual Add\nGPU: 8-15', shape='ellipse', fillcolor='lightgreen')

    # Layer 3
    with c.subgraph(name='cluster_layer3') as layer:
        layer.attr(label='Layer 3')
        
        layer.node('ln6', 'LayerNorm 6\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: 8-15', fillcolor='lightyellow')
        
        for i in range(8, 16):
            layer.node(f'qkv3_{i}', f'QKV Projection {i}\nGPU: {i}', fillcolor='lightcoral')
            layer.node(f'attn3_{i}', f'Attention {i}\nGPU: {i}', fillcolor='lightcoral')
            layer.node(f'out3_{i}', f'Output Projection {i}\nGPU: {i}', fillcolor='lightcoral')
        
        layer.node('ar6', 'All-Reduce\nGPU: 8-15', shape='parallelogram', fillcolor='orange')
        layer.node('res6', 'Residual Add\nGPU: 8-15', shape='ellipse', fillcolor='lightgreen')
        
        layer.node('ln7', 'LayerNorm 7\nGPU: 8-15', fillcolor='lightyellow')
        
        for i in range(8, 16):
            layer.node(f'mlp1_3_{i}', f'MLP Linear 4h {i}\nGPU: {i}', fillcolor='lightblue')
            layer.node(f'gelu3_{i}', f'GELU {i}\nGPU: {i}', fillcolor='lightblue')
            layer.node(f'mlp2_3_{i}', f'MLP Linear h {i}\nGPU: {i}', fillcolor='lightblue')
        
        layer.node('ar7', 'All-Reduce\nGPU: 8-15', shape='parallelogram', fillcolor='orange')
        layer.node('res7', 'Residual Add\nGPU: 8-15', shape='ellipse', fillcolor='lightgreen')

# Global output
with dot.subgraph(name='cluster_output') as c:
    c.attr(label='Global Output', style='dashed')
    c.node('final_ln', 'Final LayerNorm\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: 8-15', fillcolor='lightyellow')
    c.node('output', 'Output Projection\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, vocab=50257]\nGPU: 8-15', shape='ellipse', fillcolor='lightgreen')

# Connect the graph
# Input connections
dot.edge('input_tokens', 'embedding')
dot.edge('embedding', 'position_embed')

# Layer 0 connections
dot.edge('position_embed', 'ln0')
for i in range(8):
    dot.edge('ln0', f'qkv0_{i}')
    dot.edge(f'qkv0_{i}', f'attn0_{i}')
    dot.edge(f'attn0_{i}', f'out0_{i}')
    dot.edge(f'out0_{i}', 'ar0')

dot.edge('ar0', 'res0')
dot.edge('position_embed', 'res0')
dot.edge('res0', 'ln1')

# Layer 0 MLP connections
for i in range(8):
    dot.edge('ln1', f'mlp1_0_{i}')
    dot.edge(f'mlp1_0_{i}', f'gelu0_{i}')
    dot.edge(f'gelu0_{i}', f'mlp2_0_{i}')
    dot.edge(f'mlp2_0_{i}', 'ar1')

dot.edge('ar1', 'res1')
dot.edge('res0', 'res1')

# Layer 1 connections
dot.edge('res1', 'ln2')
for i in range(8):
    dot.edge('ln2', f'qkv1_{i}')
    dot.edge(f'qkv1_{i}', f'attn1_{i}')
    dot.edge(f'attn1_{i}', f'out1_{i}')
    dot.edge(f'out1_{i}', 'ar2')

dot.edge('ar2', 'res2')
dot.edge('res1', 'res2')
dot.edge('res2', 'ln3')

# Layer 1 MLP connections
for i in range(8):
    dot.edge('ln3', f'mlp1_1_{i}')
    dot.edge(f'mlp1_1_{i}', f'gelu1_{i}')
    dot.edge(f'gelu1_{i}', f'mlp2_1_{i}')
    dot.edge(f'mlp2_1_{i}', 'ar3')

dot.edge('ar3', 'res3')
dot.edge('res2', 'res3')

# Pipeline stage connection
dot.edge('res3', 'pipe_send')
dot.edge('pipe_send', 'ln4')

# Layer 2 connections
dot.edge('pipe_send', 'ln4')
for i in range(8, 16):
    dot.edge('ln4', f'qkv2_{i}')
    dot.edge(f'qkv2_{i}', f'attn2_{i}')
    dot.edge(f'attn2_{i}', f'out2_{i}')
    dot.edge(f'out2_{i}', 'ar4')

dot.edge('ar4', 'res4')
dot.edge('pipe_send', 'res4')
dot.edge('res4', 'ln5')

# Layer 2 MLP connections
for i in range(8, 16):
    dot.edge('ln5', f'mlp1_2_{i}')
    dot.edge(f'mlp1_2_{i}', f'gelu2_{i}')
    dot.edge(f'gelu2_{i}', f'mlp2_2_{i}')
    dot.edge(f'mlp2_2_{i}', 'ar5')

dot.edge('ar5', 'res5')
dot.edge('res4', 'res5')

# Layer 3 connections
dot.edge('res5', 'ln6')
for i in range(8, 16):
    dot.edge('ln6', f'qkv3_{i}')
    dot.edge(f'qkv3_{i}', f'attn3_{i}')
    dot.edge(f'attn3_{i}', f'out3_{i}')
    dot.edge(f'out3_{i}', 'ar6')

dot.edge('ar6', 'res6')
dot.edge('res5', 'res6')
dot.edge('res6', 'ln7')

# Layer 3 MLP connections
for i in range(8, 16):
    dot.edge('ln7', f'mlp1_3_{i}')
    dot.edge(f'mlp1_3_{i}', f'gelu3_{i}')
    dot.edge(f'gelu3_{i}', f'mlp2_3_{i}')
    dot.edge(f'mlp2_3_{i}', 'ar7')

dot.edge('ar7', 'res7')
dot.edge('res6', 'res7')

# Final connections
dot.edge('res7', 'final_ln')
dot.edge('final_ln', 'output')

# Save the DAG
dot.render('/home/wzc/data/file-share/2025-09-15-14-08-33/baseline_dense_transformer', format='svg', cleanup=False)
dot.save('/home/wzc/data/file-share/2025-09-15-14-08-33/baseline_dense_transformer.dot')

print("Baseline DAG generated successfully")
print("Files saved:")
print("- /home/wzc/data/file-share/2025-09-15-14-08-33/baseline_dense_transformer.svg")
print("- /home/wzc/data/file-share/2025-09-15-14-08-33/baseline_dense_transformer.dot")