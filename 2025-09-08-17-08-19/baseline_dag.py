import graphviz

# Create a new directed graph for baseline (TP=8, PP=2)
dot = graphviz.Digraph(comment='Baseline DAG (TP=8, PP=2)', format='svg')
dot.attr(rankdir='TB', size='20,20')

# Define model dimensions
batch_size = 1024
hidden_size = 8192
ffn_hidden = 32768
num_heads = 16
head_dim = 512
seq_len = 'seq_len'  # Variable sequence length

# Tensor parallelism splits
TP = 8
PP = 2

# Each pipeline stage has 8 layers (16 total layers / 2 stages)
layers_per_stage = 8

# GPU assignments: Stage 0 on GPUs 0-7, Stage 1 on GPUs 8-15
# Within each stage, tensor parallelism splits across 8 GPUs

# Input node
dot.node('input', f'Total Input\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: all', 
         shape='ellipse', style='filled', fillcolor='lightblue')

# Function to add attention block
def add_attention_block(stage_id, layer_id, start_node):
    layer_prefix = f's{stage_id}_l{layer_id}'
    
    # Layer norm (replicated across TP group)
    dot.node(f'{layer_prefix}_ln1', f'LayerNorm\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {stage_id*8}-{stage_id*8+7}', 
             shape='rectangle', style='filled', fillcolor='lightyellow')
    dot.edge(start_node, f'{layer_prefix}_ln1')
    
    # QKV projection (column parallel)
    qkv_output = hidden_size // TP
    dot.node(f'{layer_prefix}_qkv', f'QKV Projection\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {qkv_output*3})\\nGPU: {stage_id*8}-{stage_id*8+7}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_ln1', f'{layer_prefix}_qkv')
    
    # Reshape for attention
    dot.node(f'{layer_prefix}_reshape', f'Reshape QKV\\nINPUT: ({batch_size}, {seq_len}, {qkv_output*3})\\nOUTPUT: ({batch_size}, {num_heads//TP}, {seq_len}, {head_dim})\\nGPU: {stage_id*8}-{stage_id*8+7}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_qkv', f'{layer_prefix}_reshape')
    
    # Attention computation
    dot.node(f'{layer_prefix}_attn', f'Multi-Head Attention\\nINPUT: ({batch_size}, {num_heads//TP}, {seq_len}, {head_dim})\\nOUTPUT: ({batch_size}, {num_heads//TP}, {seq_len}, {head_dim})\\nGPU: {stage_id*8}-{stage_id*8+7}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_reshape', f'{layer_prefix}_attn')
    
    # Reshape back
    dot.node(f'{layer_prefix}_reshape_back', f'Reshape Output\\nINPUT: ({batch_size}, {num_heads//TP}, {seq_len}, {head_dim})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size//TP})\\nGPU: {stage_id*8}-{stage_id*8+7}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_attn', f'{layer_prefix}_reshape_back')
    
    # Output projection (row parallel)
    dot.node(f'{layer_prefix}_out_proj', f'Output Projection\\nINPUT: ({batch_size}, {seq_len}, {hidden_size//TP})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {stage_id*8}-{stage_id*8+7}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_reshape_back', f'{layer_prefix}_out_proj')
    
    # All-reduce for attention output
    dot.node(f'{layer_prefix}_attn_allreduce', f'All-Reduce\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {stage_id*8}-{stage_id*8+7}', 
             shape='parallelogram', style='filled', fillcolor='orange')
    dot.edge(f'{layer_prefix}_out_proj', f'{layer_prefix}_attn_allreduce')
    
    # Residual connection
    dot.node(f'{layer_prefix}_res1', f'Residual Add\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {stage_id*8}-{stage_id*8+7}', 
             shape='rectangle', style='filled', fillcolor='lightcoral')
    dot.edge(start_node, f'{layer_prefix}_res1')
    dot.edge(f'{layer_prefix}_attn_allreduce', f'{layer_prefix}_res1')
    
    return f'{layer_prefix}_res1'

# Function to add MLP block
def add_mlp_block(stage_id, layer_id, start_node):
    layer_prefix = f's{stage_id}_l{layer_id}'
    
    # Layer norm (replicated)
    dot.node(f'{layer_prefix}_ln2', f'LayerNorm\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {stage_id*8}-{stage_id*8+7}', 
             shape='rectangle', style='filled', fillcolor='lightyellow')
    dot.edge(start_node, f'{layer_prefix}_ln2')
    
    # First linear (column parallel)
    ffn_split = ffn_hidden // TP
    dot.node(f'{layer_prefix}_fc1', f'FC1 (Column Parallel)\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {ffn_split})\\nGPU: {stage_id*8}-{stage_id*8+7}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_ln2', f'{layer_prefix}_fc1')
    
    # GELU activation
    dot.node(f'{layer_prefix}_gelu', f'GELU\\nINPUT: ({batch_size}, {seq_len}, {ffn_split})\\nOUTPUT: ({batch_size}, {seq_len}, {ffn_split})\\nGPU: {stage_id*8}-{stage_id*8+7}', 
             shape='rectangle', style='filled', fillcolor='lightyellow')
    dot.edge(f'{layer_prefix}_fc1', f'{layer_prefix}_gelu')
    
    # Second linear (row parallel)
    dot.node(f'{layer_prefix}_fc2', f'FC2 (Row Parallel)\\nINPUT: ({batch_size}, {seq_len}, {ffn_split})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {stage_id*8}-{stage_id*8+7}', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge(f'{layer_prefix}_gelu', f'{layer_prefix}_fc2')
    
    # All-reduce for MLP output
    dot.node(f'{layer_prefix}_mlp_allreduce', f'All-Reduce\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {stage_id*8}-{stage_id*8+7}', 
             shape='parallelogram', style='filled', fillcolor='orange')
    dot.edge(f'{layer_prefix}_fc2', f'{layer_prefix}_mlp_allreduce')
    
    # Residual connection
    dot.node(f'{layer_prefix}_res2', f'Residual Add\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: {stage_id*8}-{stage_id*8+7}', 
             shape='rectangle', style='filled', fillcolor='lightcoral')
    dot.edge(start_node, f'{layer_prefix}_res2')
    dot.edge(f'{layer_prefix}_mlp_allreduce', f'{layer_prefix}_res2')
    
    return f'{layer_prefix}_res2'

# Build pipeline stage 0 (layers 0-7)
prev_node = 'input'
for layer in range(8):
    attn_out = add_attention_block(0, layer, prev_node)
    mlp_out = add_mlp_block(0, layer, attn_out)
    prev_node = mlp_out

# Pipeline communication between stages
dot.node('pipeline_send', f'Pipeline Send\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nOUTPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: 7→8', 
         shape='parallelogram', style='filled', fillcolor='purple')
dot.edge(prev_node, 'pipeline_send')

# Build pipeline stage 1 (layers 8-15)
prev_node = 'pipeline_send'
for layer in range(8, 16):
    attn_out = add_attention_block(1, layer, prev_node)
    mlp_out = add_mlp_block(1, layer, attn_out)
    prev_node = mlp_out

# Output node
dot.node('output', f'Total Output\\nINPUT: ({batch_size}, {seq_len}, {hidden_size})\\nGPU: all', 
         shape='ellipse', style='filled', fillcolor='lightblue')
dot.edge(prev_node, 'output')

# Save the DAG
dot.render('/home/wzc/data/file-share/2025-09-08-17-08-19/baseline_dag')

print("Baseline DAG generated successfully!")