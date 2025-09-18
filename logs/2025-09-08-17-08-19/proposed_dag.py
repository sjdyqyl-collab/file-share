import graphviz

# Create a new directed graph for proposed layer-wise deployment
dot = graphviz.Digraph(comment='Proposed Layer-wise DAG', format='svg')
dot.attr(rankdir='TB', size='30,30')

# Define model dimensions
batch_size = 1024
hidden_size = 8192
ffn_hidden = 32768
num_heads = 16
head_dim = 512
seq_len = 'seq_len'  # Variable sequence length

# Input node
dot.node('input', f'Total Input\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: 0', 
         shape='ellipse', style='filled', fillcolor='lightblue')

# Function to add complete layer on single GPU
def add_layer(layer_id, gpu_id, start_node):
    layer_prefix = f'layer{layer_id}'
    
    # Layer norm 1
    dot.node(f'{layer_prefix}_ln1', f'LayerNorm\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightyellow')
    dot.edge(start_node, f'{layer_prefix}_ln1')
    
    # Q projection
    dot.node(f'{layer_prefix}_q', f'Q Projection\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_ln1', f'{layer_prefix}_q')
    
    # K projection
    dot.node(f'{layer_prefix}_k', f'K Projection\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_ln1', f'{layer_prefix}_k')
    
    # V projection
    dot.node(f'{layer_prefix}_v', f'V Projection\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_ln1', f'{layer_prefix}_v')
    
    # Reshape Q
    dot.node(f'{layer_prefix}_q_reshape', f'Reshape Q\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {num_heads}, {seq_len}, {head_dim})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_q', f'{layer_prefix}_q_reshape')
    
    # Reshape K
    dot.node(f'{layer_prefix}_k_reshape', f'Reshape K\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {num_heads}, {seq_len}, {head_dim})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_k', f'{layer_prefix}_k_reshape')
    
    # Reshape V
    dot.node(f'{layer_prefix}_v_reshape', f'Reshape V\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {num_heads}, {seq_len}, {head_dim})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_v', f'{layer_prefix}_v_reshape')
    
    # Attention scores
    dot.node(f'{layer_prefix}_attn_scores', f'Attention Scores\\nINPUT: ({batch_size}, {num_heads}, {seq_len}, {head_dim})\\nOUTPUT: ({batch_size}, {num_heads}, {seq_len}, {seq_len})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_q_reshape', f'{layer_prefix}_attn_scores')
    dot.edge(f'{layer_prefix}_k_reshape', f'{layer_prefix}_attn_scores')
    
    # Softmax
    dot.node(f'{layer_prefix}_softmax', f'Softmax\\nINPUT: ({batch_size}, {num_heads}, {seq_len}, {seq_len})\\nOUTPUT: ({batch_size}, {num_heads}, {seq_len}, {seq_len})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightyellow')
    dot.edge(f'{layer_prefix}_attn_scores', f'{layer_prefix}_softmax')
    
    # Attention output
    dot.node(f'{layer_prefix}_attn_out', f'Attention Output\\nINPUT: ({batch_size}, {num_heads}, {seq_len}, {seq_len})\\nOUTPUT: ({batch_size}, {num_heads}, {seq_len}, {head_dim})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_softmax', f'{layer_prefix}_attn_out')
    dot.edge(f'{layer_prefix}_v_reshape', f'{layer_prefix}_attn_out')
    
    # Reshape attention output
    dot.node(f'{layer_prefix}_attn_reshape', f'Reshape Attention\\nINPUT: ({batch_size}, {num_heads}, {seq_len}, {head_dim})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_attn_out', f'{layer_prefix}_attn_reshape')
    
    # Output projection
    dot.node(f'{layer_prefix}_out_proj', f'Output Projection\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_attn_reshape', f'{layer_prefix}_out_proj')
    
    # Residual connection 1
    dot.node(f'{layer_prefix}_res1', f'Residual Add\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightcoral')
    dot.edge(start_node, f'{layer_prefix}_res1')
    dot.edge(f'{layer_prefix}_out_proj', f'{layer_prefix}_res1')
    
    # Layer norm 2
    dot.node(f'{layer_prefix}_ln2', f'LayerNorm\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightyellow')
    dot.edge(f'{layer_prefix}_res1', f'{layer_prefix}_ln2')
    
    # FC1
    dot.node(f'{layer_prefix}_fc1', f'FC1\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {ffn_hidden})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_ln2', f'{layer_prefix}_fc1')
    
    # GELU
    dot.node(f'{layer_prefix}_gelu', f'GELU\\nINPUT: ({batch_size}, {seq_len}, {ffn_hidden})\\nOUTPUT: ({batch_size}, {seq_len}, {ffn_hidden})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightyellow')
    dot.edge(f'{layer_prefix}_fc1', f'{layer_prefix}_gelu')
    
    # FC2
    dot.node(f'{layer_prefix}_fc2', f'FC2\\nINPUT: ({batch_size}, {seq_len}, {ffn_hidden})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_gelu', f'{layer_prefix}_fc2')
    
    # Residual connection 2
    dot.node(f'{layer_prefix}_res2', f'Residual Add\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {gpu_id}', 
             shape='rectangle', style='filled', fillcolor='lightcoral')
    dot.edge(f'{layer_prefix}_res1', f'{layer_prefix}_res2')
    dot.edge(f'{layer_prefix}_fc2', f'{layer_prefix}_res2')
    
    # Inter-GPU communication
    if gpu_id < 15:
        dot.node(f'{layer_prefix}_send', f'Inter-GPU Transfer\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {gpu_id}→{gpu_id+1}', 
                 shape='parallelogram', style='filled', fillcolor='purple')
        dot.edge(f'{layer_prefix}_res2', f'{layer_prefix}_send')
        return f'{layer_prefix}_send'
    else:
        return f'{layer_prefix}_res2'

# Build 16 layers across 16 GPUs
prev_node = 'input'
for layer_id in range(16):
    prev_node = add_layer(layer_id, layer_id, prev_node)

# Output node
dot.node('output', f'Total Output\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: 15', 
         shape='ellipse', style='filled', fillcolor='lightblue')
dot.edge(prev_node, 'output')

# Save the DAG
dot.render('/home/wzc/data/file-share/2025-09-08-17-08-19/proposed_dag')

print("Proposed layer-wise DAG generated successfully!")