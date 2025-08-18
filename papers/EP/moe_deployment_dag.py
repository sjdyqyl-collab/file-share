import graphviz

# Create a new directed graph
dot = graphviz.Digraph('MoE_Cross_Node_Expert_Parallelism', 
                       comment='Large-scale Cross-Node Expert Parallelism DAG',
                       graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'nodesep': '0.8', 'ranksep': '1.2'})

# Define colors for different components
color_map = {
    'input': 'lightblue',
    'attention': 'lightgreen',
    'expert': 'lightyellow',
    'communication': 'lightcoral',
    'output': 'lightpink'
}

# Global dimensions
batch_size = 1024
seq_len = 512  # Assuming standard sequence length
hidden_dim = 4096  # Standard for large models
num_heads = 32
head_dim = hidden_dim // num_heads  # 128
expert_dim = 16384  # Expert FFN hidden dimension

# Add input node
dot.node('input', f'Input\\nBatch={batch_size}, Seq={seq_len}, Hidden={hidden_dim}', 
         shape='box', style='filled', fillcolor=color_map['input'])

# Add preprocessing (LayerNorm)
dot.node('ln1', f'LayerNorm\\nShape: [{batch_size}, {seq_len}, {hidden_dim}]', 
         shape='ellipse', style='filled', fillcolor=color_map['attention'])
dot.edge('input', 'ln1')

# Add attention mechanism for each layer
for layer in range(4):
    # Multi-head attention
    dot.node(f'attn_qkv_{layer}', f'Layer {layer} QKV Linear\\nTP=8 Column Parallel\\nShape: [{batch_size}, {seq_len}, {hidden_dim}] → [{batch_size}, {seq_len}, {hidden_dim*3}]', 
             shape='ellipse', style='filled', fillcolor=color_map['attention'])
    dot.node(f'attn_split_{layer}', f'Split QKV\\nShape: Q[{batch_size}, {seq_len}, {hidden_dim}], K/V same', 
             shape='ellipse', style='filled', fillcolor=color_map['attention'])
    dot.node(f'attn_score_{layer}', f'Attention Scores\\nShape: [{batch_size}, {num_heads}, {seq_len}, {seq_len}]', 
             shape='ellipse', style='filled', fillcolor=color_map['attention'])
    dot.node(f'attn_out_{layer}', f'Attention Output Linear\\nTP=8 Row Parallel\\nShape: [{batch_size}, {seq_len}, {hidden_dim}]', 
             shape='ellipse', style='filled', fillcolor=color_map['attention'])
    dot.node(f'attn_residual_{layer}', f'Residual Add\\nShape: [{batch_size}, {seq_len}, {hidden_dim}]', 
             shape='ellipse', style='filled', fillcolor=color_map['attention'])
    
    # Add LayerNorm after attention
    dot.node(f'ln2_{layer}', f'Post-Attention LayerNorm\\nShape: [{batch_size}, {seq_len}, {hidden_dim}]', 
             shape='ellipse', style='filled', fillcolor=color_map['attention'])
    
    # Add MoE layer with experts distributed across GPUs
    dot.node(f'moe_gate_{layer}', f'MoE Gate\\nShape: [{batch_size*seq_len}, {hidden_dim}] → Top-2 routing', 
             shape='diamond', style='filled', fillcolor=color_map['expert'])
    
    # Create expert nodes for this layer (64 experts distributed across 64 GPUs)
    for gpu_id in range(64):
        expert_id = layer * 64 + gpu_id
        # Each GPU has 1 expert from this layer
        dot.node(f'expert_{layer}_{gpu_id}', 
                 f'Expert {expert_id}\\nGPU {gpu_id}\\nShape: [{batch_size*seq_len}, {hidden_dim}] → [{batch_size*seq_len}, {expert_dim}] → [{batch_size*seq_len}, {hidden_dim}]', 
                 shape='box', style='filled', fillcolor=color_map['expert'])
        
        # Add communication path from gate to expert
        dot.node(f'comm_gate_expert_{layer}_{gpu_id}', 
                 f'Token Routing\\nGPU 0→{gpu_id}\\nShape: Tokens × {hidden_dim}', 
                 shape='parallelogram', style='filled', fillcolor=color_map['communication'])
        dot.edge(f'moe_gate_{layer}', f'comm_gate_expert_{layer}_{gpu_id}')
        dot.edge(f'comm_gate_expert_{layer}_{gpu_id}', f'expert_{layer}_{gpu_id}')
        
        # Add communication path back from expert
        dot.node(f'comm_expert_agg_{layer}_{gpu_id}', 
                 f'Expert Output\\nGPU {gpu_id}→0\\nShape: Tokens × {hidden_dim}', 
                 shape='parallelogram', style='filled', fillcolor=color_map['communication'])
        dot.edge(f'expert_{layer}_{gpu_id}', f'comm_expert_agg_{layer}_{gpu_id}')
    
    # Add expert aggregation
    dot.node(f'expert_agg_{layer}', f'Expert Output Aggregation\\nAll-Reduce across GPUs\\nShape: [{batch_size*seq_len}, {hidden_dim}]', 
             shape='hexagon', style='filled', fillcolor=color_map['expert'])
    
    # Connect all expert outputs to aggregation
    for gpu_id in range(64):
        dot.edge(f'comm_expert_agg_{layer}_{gpu_id}', f'expert_agg_{layer}')
    
    # Residual connection after MoE
    dot.node(f'moe_residual_{layer}', f'MoE Residual Add\\nShape: [{batch_size}, {seq_len}, {hidden_dim}]', 
             shape='ellipse', style='filled', fillcolor=color_map['expert'])
    
    # Connect layer components
    if layer == 0:
        dot.edge('ln1', 'attn_qkv_0')
    else:
        dot.edge(f'moe_residual_{layer-1}', f'attn_qkv_{layer}')
    
    dot.edge(f'attn_qkv_{layer}', f'attn_split_{layer}')
    dot.edge(f'attn_split_{layer}', f'attn_score_{layer}')
    dot.edge(f'attn_score_{layer}', f'attn_out_{layer}')
    dot.edge(f'attn_out_{layer}', f'attn_residual_{layer}')
    dot.edge(f'attn_residual_{layer}', f'ln2_{layer}')
    dot.edge(f'ln2_{layer}', f'moe_gate_{layer}')
    dot.edge(f'expert_agg_{layer}', f'moe_residual_{layer}')

# Add final output
final_shape = f'[{batch_size}, {seq_len}, {hidden_dim}]'
dot.node('output', f'Final Output\\n{final_shape}', 
         shape='box', style='filled', fillcolor=color_map['output'])
dot.edge('moe_residual_3', 'output')

# Add subgraphs to group GPUs
for gpu_id in range(64):
    with dot.subgraph(name=f'cluster_gpu_{gpu_id}') as c:
        c.attr(label=f'GPU {gpu_id}', style='dashed', color='blue')
        # Add all experts on this GPU
        for layer in range(4):
            c.node(f'expert_{layer}_{gpu_id}')

# Save the DAG
dot.render('/home/wzc/data/papers/EP/moe_deployment_dag', format='dot', cleanup=False)
dot.render('/home/wzc/data/papers/EP/moe_deployment_dag', format='png', cleanup=False)

# Also generate a simplified version showing GPU boundaries more clearly
dot_simple = graphviz.Digraph('MoE_Simplified_DAG', 
                              comment='Simplified MoE Deployment View',
                              graph_attr={'rankdir': 'LR', 'splines': 'ortho'})

# Show high-level flow
dot_simple.node('input_tokens', f'Input Tokens\\n{batch_size}×{seq_len}×{hidden_dim}', 
                shape='box', style='filled', fillcolor=color_map['input'])

for layer in range(4):
    dot_simple.node(f'layer_{layer}', f'Layer {layer}\\n64 Experts across 64 GPUs\\n1 Expert per GPU', 
                    shape='component', style='filled', fillcolor=color_map['expert'])
    
    # Show communication
    dot_simple.node(f'comm_{layer}', f'Cross-GPU Communication\\nToken Routing & Aggregation', 
                    shape='parallelogram', style='filled', fillcolor=color_map['communication'])
    
    if layer == 0:
        dot_simple.edge('input_tokens', f'layer_{layer}')
    else:
        dot_simple.edge(f'comm_{layer-1}', f'layer_{layer}')
    
    dot_simple.edge(f'layer_{layer}', f'comm_{layer}')

dot_simple.node('output_tokens', f'Output Tokens\\n{batch_size}×{seq_len}×{hidden_dim}', 
                shape='box', style='filled', fillcolor=color_map['output'])
dot_simple.edge('comm_3', 'output_tokens')

# Save simplified DAG
dot_simple.render('/home/wzc/data/papers/EP/moe_simplified_dag', format='dot', cleanup=False)

print("DAG files generated successfully:")
print("- /home/wzc/data/papers/EP/moe_deployment_dag.dot (complete detailed DAG)")
print("- /home/wzc/data/papers/EP/moe_simplified_dag.dot (simplified view)")