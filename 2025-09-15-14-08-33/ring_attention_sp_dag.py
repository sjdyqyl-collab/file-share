import graphviz

# Create Ring Attention + Sequence Parallelism DAG for 16 GPUs
dot = graphviz.Digraph('ring_attention_sp_transformer', comment='Ring Attention + Sequence Parallelism (16 GPUs)')
dot.attr(rankdir='TB', size='30,30')

# Define node styles
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
dot.attr('edge', fontsize='10')

# Global input - sequence split across 16 devices
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='Global Input - Sequence Split', style='dashed')
    c.node('input_tokens', 'Input Tokens\nInput: [batch_size=1024, seq_len=10000]\nGPU: all GPUs', shape='ellipse', fillcolor='lightgreen')
    
    # Sequence split across 16 devices
    for i in range(16):
        c.node(f'split_{i}', f'Sequence Split {i}\nInput: [batch_size=1024, seq_len=10000]\nOutput: [batch_size=1024, seq_len=625, hidden=8192]\nGPU: {i}', shape='parallelogram', fillcolor='orange')
    
    # Embedding on each device
    for i in range(16):
        c.node(f'embed_{i}', f'Token Embedding {i}\nInput: [batch_size=1024, seq_len=625]\nOutput: [batch_size=1024, seq_len=625, hidden=8192]\nGPU: {i}', shape='rectangle')
    
    # Position embedding on each device
    for i in range(16):
        c.node(f'pos_embed_{i}', f'Position Embedding {i}\nInput: [batch_size=1024, seq_len=625, hidden=8192]\nOutput: [batch_size=1024, seq_len=625, hidden=8192]\nGPU: {i}', shape='rectangle')

# Process each layer with Ring Attention
for layer_idx in range(4):
    with dot.subgraph(name=f'cluster_layer{layer_idx}') as layer_cluster:
        layer_cluster.attr(label=f'Layer {layer_idx} - Ring Attention + Sequence Parallel', style='rounded')
        
        # LayerNorm for each device
        for i in range(16):
            layer_cluster.node(f'ln{layer_idx}_{i}', f'LayerNorm {layer_idx}.{i}\nInput: [batch_size=1024, seq_len=625, hidden=8192]\nOutput: [batch_size=1024, seq_len=625, hidden=8192]\nGPU: {i}', fillcolor='lightyellow')
        
        # QKV projections for each device (no tensor parallelism)
        for i in range(16):
            layer_cluster.node(f'q_proj{layer_idx}_{i}', f'Query Projection {layer_idx}.{i}\nInput: [batch_size=1024, seq_len=625, hidden=8192]\nOutput: [batch_size=1024, seq_len=625, qkv=8192]\nGPU: {i}', fillcolor='lightcoral')
            layer_cluster.node(f'k_proj{layer_idx}_{i}', f'Key Projection {layer_idx}.{i}\nInput: [batch_size=1024, seq_len=625, hidden=8192]\nOutput: [batch_size=1024, seq_len=625, qkv=8192]\nGPU: {i}', fillcolor='lightcoral')
            layer_cluster.node(f'v_proj{layer_idx}_{i}', f'Value Projection {layer_idx}.{i}\nInput: [batch_size=1024, seq_len=625, hidden=8192]\nOutput: [batch_size=1024, seq_len=625, qkv=8192]\nGPU: {i}', fillcolor='lightcoral')
        
        # Ring Attention computation
        for stage in range(16):
            for device in range(16):
                src_device = (device - stage) % 16
                layer_cluster.node(f'ring_attn{layer_idx}_{stage}_{device}', 
                                 f'Ring Attention Stage {stage}.{device}\nInput: Q=[1024,625,8192], K=[1024,625,8192], V=[1024,625,8192]\nOutput: [batch_size=1024, seq_len=625, hidden=8192]\nGPU: {device} (from {src_device})', 
                                 fillcolor='lightcoral')
        
        # Output projection for each device
        for i in range(16):
            layer_cluster.node(f'out_proj{layer_idx}_{i}', f'Output Projection {layer_idx}.{i}\nInput: [batch_size=1024, seq_len=625, hidden=8192]\nOutput: [batch_size=1024, seq_len=625, hidden=8192]\nGPU: {i}', fillcolor='lightcoral')
        
        # Residual connections
        for i in range(16):
            layer_cluster.node(f'res_attn{layer_idx}_{i}', f'Residual Add Attention {layer_idx}.{i}\nInput: [batch_size=1024, seq_len=625, hidden=8192]×2\nOutput: [batch_size=1024, seq_len=625, hidden=8192]\nGPU: {i}', shape='ellipse', fillcolor='lightgreen')
        
        # LayerNorm for MLP
        for i in range(16):
            layer_cluster.node(f'ln_mlp{layer_idx}_{i}', f'LayerNorm MLP {layer_idx}.{i}\nInput: [batch_size=1024, seq_len=625, hidden=8192]\nOutput: [batch_size=1024, seq_len=625, hidden=8192]\nGPU: {i}', fillcolor='lightyellow')
        
        # MLP components (sequence parallel)
        for i in range(16):
            layer_cluster.node(f'mlp1_{layer_idx}_{i}', f'MLP Linear 4h {layer_idx}.{i}\nInput: [batch_size=1024, seq_len=625, hidden=8192]\nOutput: [batch_size=1024, seq_len=625, ffn=32768]\nGPU: {i}', fillcolor='lightblue')
            layer_cluster.node(f'gelu_{layer_idx}_{i}', f'GELU {layer_idx}.{i}\nInput: [batch_size=1024, seq_len=625, ffn=32768]\nOutput: [batch_size=1024, seq_len=625, ffn=32768]\nGPU: {i}', fillcolor='lightblue')
            layer_cluster.node(f'mlp2_{layer_idx}_{i}', f'MLP Linear h {layer_idx}.{i}\nInput: [batch_size=1024, seq_len=625, ffn=32768]\nOutput: [batch_size=1024, seq_len=625, hidden=8192]\nGPU: {i}', fillcolor='lightblue')
        
        # Residual connections for MLP
        for i in range(16):
            layer_cluster.node(f'res_mlp{layer_idx}_{i}', f'Residual Add MLP {layer_idx}.{i}\nInput: [batch_size=1024, seq_len=625, hidden=8192]×2\nOutput: [batch_size=1024, seq_len=625, hidden=8192]\nGPU: {i}', shape='ellipse', fillcolor='lightgreen')

# Communication nodes for Ring Attention
with dot.subgraph(name='cluster_communication') as c:
    c.attr(label='Ring Communication', style='dashed')
    
    for layer_idx in range(4):
        for stage in range(16):
            for device in range(16):
                next_device = (device + 1) % 16
                c.node(f'send_kv{layer_idx}_{stage}_{device}', 
                      f'Send KV Block\nStage {stage}.{device} → {next_device}\nData: [batch_size=1024, seq_len=625, hidden=8192×2]\nGPU: {device} → {next_device}', 
                      shape='parallelogram', fillcolor='orange')
                
                c.node(f'recv_kv{layer_idx}_{stage}_{device}', 
                      f'Receive KV Block\nStage {stage}.{device} ← {(device-1)%16}\nData: [batch_size=1024, seq_len=625, hidden=8192×2]\nGPU: {(device-1)%16} → {device}', 
                      shape='parallelogram', fillcolor='orange')

# Global output - sequence gather
with dot.subgraph(name='cluster_output') as c:
    c.attr(label='Global Output - Sequence Gather', style='dashed')
    
    # Gather from all devices
    for i in range(16):
        c.node(f'gather_{i}', f'Gather Sequence {i}\nInput: [batch_size=1024, seq_len=625, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: {i} → all', shape='parallelogram', fillcolor='orange')
    
    c.node('final_ln', 'Final LayerNorm\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, hidden=8192]\nGPU: all GPUs', fillcolor='lightyellow')
    c.node('output', 'Output Projection\nInput: [batch_size=1024, seq_len=10000, hidden=8192]\nOutput: [batch_size=1024, seq_len=10000, vocab=50257]\nGPU: all GPUs', shape='ellipse', fillcolor='lightgreen')

# Connect the graph
# Input connections
dot.edge('input_tokens', 'split_0')
for i in range(16):
    dot.edge('input_tokens', f'split_{i}')
    dot.edge(f'split_{i}', f'embed_{i}')
    dot.edge(f'embed_{i}', f'pos_embed_{i}')

# Layer 0 connections
for i in range(16):
    dot.edge(f'pos_embed_{i}', f'ln0_{i}')
    dot.edge(f'ln0_{i}', f'q_proj0_{i}')
    dot.edge(f'ln0_{i}', f'k_proj0_{i}')
    dot.edge(f'ln0_{i}', f'v_proj0_{i}')

# Ring Attention connections for layer 0
for device in range(16):
    dot.edge(f'q_proj0_{device}', f'ring_attn0_0_{device}')
    dot.edge(f'k_proj0_{device}', f'ring_attn0_0_{device}')
    dot.edge(f'v_proj0_{device}', f'ring_attn0_0_{device}')
    
    for stage in range(15):
        # Send KV blocks
        dot.edge(f'ring_attn0_{stage}_{device}', f'send_kv0_{stage}_{device}')
        dot.edge(f'send_kv0_{stage}_{device}', f'recv_kv0_{stage+1}_{(device+1)%16}')
        dot.edge(f'recv_kv0_{stage+1}_{device}', f'ring_attn0_{stage+1}_{device}')
    
    # Final attention output
    dot.edge(f'ring_attn0_15_{device}', f'out_proj0_{device}')
    dot.edge(f'out_proj0_{device}', f'res_attn0_{device}')
    dot.edge(f'pos_embed_{device}', f'res_attn0_{device}')

# Layer 0 MLP connections
for i in range(16):
    dot.edge(f'res_attn0_{i}', f'ln_mlp0_{i}')
    dot.edge(f'ln_mlp0_{i}', f'mlp1_0_{i}')
    dot.edge(f'mlp1_0_{i}', f'gelu_0_{i}')
    dot.edge(f'gelu_0_{i}', f'mlp2_0_{i}')
    dot.edge(f'mlp2_0_{i}', f'res_mlp0_{i}')
    dot.edge(f'res_attn0_{i}', f'res_mlp0_{i}')

# Connect layers sequentially
for layer_idx in range(1, 4):
    prev_layer = layer_idx - 1
    for i in range(16):
        dot.edge(f'res_mlp{prev_layer}_{i}', f'ln{layer_idx}_{i}')
        dot.edge(f'ln{layer_idx}_{i}', f'q_proj{layer_idx}_{i}')
        dot.edge(f'ln{layer_idx}_{i}', f'k_proj{layer_idx}_{i}')
        dot.edge(f'ln{layer_idx}_{i}', f'v_proj{layer_idx}_{i}')
        
        # Ring attention for this layer
        dot.edge(f'q_proj{layer_idx}_{i}', f'ring_attn{layer_idx}_0_{i}')
        dot.edge(f'k_proj{layer_idx}_{i}', f'ring_attn{layer_idx}_0_{i}')
        dot.edge(f'v_proj{layer_idx}_{i}', f'ring_attn{layer_idx}_0_{i}')
        
        for stage in range(15):
            dot.edge(f'ring_attn{layer_idx}_{stage}_{i}', f'send_kv{layer_idx}_{stage}_{i}')
            dot.edge(f'send_kv{layer_idx}_{stage}_{i}', f'recv_kv{layer_idx}_{stage+1}_{(i+1)%16}')
            dot.edge(f'recv_kv{layer_idx}_{stage+1}_{i}', f'ring_attn{layer_idx}_{stage+1}_{i}')
        
        dot.edge(f'ring_attn{layer_idx}_15_{i}', f'out_proj{layer_idx}_{i}')
        dot.edge(f'out_proj{layer_idx}_{i}', f'res_attn{layer_idx}_{i}')
        dot.edge(f'res_mlp{prev_layer}_{i}', f'res_attn{layer_idx}_{i}')
        
        # MLP connections
        dot.edge(f'res_attn{layer_idx}_{i}', f'ln_mlp{layer_idx}_{i}')
        dot.edge(f'ln_mlp{layer_idx}_{i}', f'mlp1_{layer_idx}_{i}')
        dot.edge(f'mlp1_{layer_idx}_{i}', f'gelu_{layer_idx}_{i}')
        dot.edge(f'gelu_{layer_idx}_{i}', f'mlp2_{layer_idx}_{i}')
        dot.edge(f'mlp2_{layer_idx}_{i}', f'res_mlp{layer_idx}_{i}')
        dot.edge(f'res_attn{layer_idx}_{i}', f'res_mlp{layer_idx}_{i}')

# Output connections
for i in range(16):
    dot.edge(f'res_mlp3_{i}', f'gather_{i}')
    dot.edge(f'gather_{i}', 'final_ln')

dot.edge('final_ln', 'output')

# Save the DAG
dot.render('/home/wzc/data/file-share/2025-09-15-14-08-33/ring_attention_sp_transformer', format='svg', cleanup=False)
dot.save('/home/wzc/data/file-share/2025-09-15-14-08-33/ring_attention_sp_transformer.dot')

print("Ring Attention + Sequence Parallelism DAG generated successfully")
print("Files saved:")
print("- /home/wzc/data/file-share/2025-09-15-14-08-33/ring_attention_sp_transformer.svg")
print("- /home/wzc/data/file-share/2025-09-15-14-08-33/ring_attention_sp_transformer.dot")