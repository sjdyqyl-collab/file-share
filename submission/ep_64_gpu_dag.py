import graphviz

# Create the DAG for the proposed cross-node expert parallelism (64 GPUs, 1 expert per GPU)
dot = graphviz.Digraph(comment='Cross-Node Expert Parallelism - 64 GPUs')
dot.attr(rankdir='TB', size='30,20')

# Define node styles
dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Communication nodes
dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation nodes
dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/aggregation nodes

# Model parameters
num_layers = 4
experts_per_layer = 16
total_gpus = 64
batch_size = 1024
seq_len = 1  # Assuming single token for simplicity
hidden_size = 8192  # 16 heads * 512 dim per head
mlp_hidden_size = 32768

# Input processing
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='Input Processing', style='dashed')
    c.node('input_tokens', f'Input Tokens\\nShape: ({batch_size}, {seq_len})\\nGPU: CPU', shape='ellipse', fillcolor='lightblue')
    c.node('embed', f'Embedding Layer\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: all GPUs', shape='rectangle', fillcolor='lightgreen')
    c.node('pos_embed', f'Positional Encoding\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: all GPUs', shape='rectangle', fillcolor='lightgreen')
    
    dot.edge('input_tokens', 'embed')
    dot.edge('embed', 'pos_embed')

# Process each layer
prev_output = 'pos_embed'
for layer_idx in range(num_layers):
    with dot.subgraph(name=f'cluster_layer_{layer_idx}') as c:
        c.attr(label=f'Layer {layer_idx + 1}', style='dashed')
        
        # MHA computation (replicated across all GPUs)
        mha_qkv = f'mha_qkv_{layer_idx}'
        mha_attn = f'mha_attn_{layer_idx}'
        mha_out = f'mha_out_{layer_idx}'
        mha_residual = f'mha_residual_{layer_idx}'
        
        c.node(mha_qkv, f'MHA QKV Linear\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: all GPUs', shape='rectangle', fillcolor='lightgreen')
        c.node(mha_attn, f'MHA Attention\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: all GPUs', shape='rectangle', fillcolor='lightgreen')
        c.node(mha_out, f'MHA Output Linear\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: all GPUs', shape='rectangle', fillcolor='lightgreen')
        c.node(mha_residual, f'MHA Residual Add\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: all GPUs', shape='parallelogram', fillcolor='lightyellow')
        
        dot.edge(prev_output, mha_qkv)
        dot.edge(mha_qkv, mha_attn)
        dot.edge(mha_attn, mha_out)
        dot.edge(mha_out, mha_residual)
        dot.edge(prev_output, mha_residual)  # Residual connection
        
        # Layer normalization after MHA
        ln1 = f'ln1_{layer_idx}'
        c.node(ln1, f'Layer Norm 1\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: all GPUs', shape='rectangle', fillcolor='lightgreen')
        dot.edge(mha_residual, ln1)
        
        # MoE layer - expert routing and computation
        gate = f'gate_{layer_idx}'
        c.node(gate, f'Expert Gate\\nShape: ({batch_size}, {seq_len}, {experts_per_layer})\\nGPU: all GPUs', shape='parallelogram', fillcolor='lightyellow')
        dot.edge(ln1, gate)
        
        # Expert computation (distributed across 16 GPUs per layer)
        expert_outputs = []
        for expert_idx in range(experts_per_layer):
            gpu_id = layer_idx * experts_per_layer + expert_idx
            
            # Token routing to expert
            route_to_expert = f'route_to_expert_{layer_idx}_{expert_idx}'
            c.node(route_to_expert, f'Route to Expert {expert_idx}\\nShape: (tokens, {hidden_size})\\nGPU: {gpu_id}', shape='ellipse', fillcolor='lightblue')
            dot.edge(gate, route_to_expert, style='dashed')
            
            # Expert computation
            expert_linear1 = f'expert_linear1_{layer_idx}_{expert_idx}'
            expert_activation = f'expert_activation_{layer_idx}_{expert_idx}'
            expert_linear2 = f'expert_linear2_{layer_idx}_{expert_idx}'
            
            c.node(expert_linear1, f'Expert {expert_idx} Linear 1\\nShape: (tokens, {mlp_hidden_size})\\nGPU: {gpu_id}', shape='rectangle', fillcolor='lightgreen')
            c.node(expert_activation, f'Expert {expert_idx} GELU\\nShape: (tokens, {mlp_hidden_size})\\nGPU: {gpu_id}', shape='rectangle', fillcolor='lightgreen')
            c.node(expert_linear2, f'Expert {expert_idx} Linear 2\\nShape: (tokens, {hidden_size})\\nGPU: {gpu_id}', shape='rectangle', fillcolor='lightgreen')
            
            dot.edge(route_to_expert, expert_linear1)
            dot.edge(expert_linear1, expert_activation)
            dot.edge(expert_activation, expert_linear2)
            
            expert_outputs.append(expert_linear2)
        
        # Aggregate expert outputs
        aggregate = f'aggregate_{layer_idx}'
        c.node(aggregate, f'Aggregate Expert Outputs\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: all GPUs', shape='parallelogram', fillcolor='lightyellow')
        
        for expert_idx, expert_out in enumerate(expert_outputs):
            dot.edge(expert_out, aggregate)
        
        # Final residual connection and layer norm
        moe_residual = f'moe_residual_{layer_idx}'
        ln2 = f'ln2_{layer_idx}'
        
        c.node(moe_residual, f'MoE Residual Add\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: all GPUs', shape='parallelogram', fillcolor='lightyellow')
        c.node(ln2, f'Layer Norm 2\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: all GPUs', shape='rectangle', fillcolor='lightgreen')
        
        dot.edge(ln1, moe_residual)
        dot.edge(aggregate, moe_residual)
        dot.edge(moe_residual, ln2)
        
        prev_output = ln2

# Output processing
with dot.subgraph(name='cluster_output') as c:
    c.attr(label='Output Processing', style='dashed')
    
    final_norm = 'final_norm'
    output_linear = 'output_linear'
    softmax = 'softmax'
    
    c.node(final_norm, f'Final Layer Norm\\nShape: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: all GPUs', shape='rectangle', fillcolor='lightgreen')
    c.node(output_linear, f'Output Linear\\nShape: ({batch_size}, {seq_len}, vocab_size)\\nGPU: all GPUs', shape='rectangle', fillcolor='lightgreen')
    c.node(softmax, f'Softmax\\nShape: ({batch_size}, {seq_len}, vocab_size)\\nGPU: all GPUs', shape='rectangle', fillcolor='lightgreen')
    
    dot.edge(prev_output, final_norm)
    dot.edge(final_norm, output_linear)
    dot.edge(output_linear, softmax)

# Save the DAG
dot.render('/home/wzc/data/file-share/submission/ep_64_gpu_dag', format='svg', cleanup=False)
print("Generated: ep_64_gpu_dag.svg")