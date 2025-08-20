import graphviz

# Create a new directed graph for proposed cross-node expert parallelism (64 GPUs, 1 expert per GPU)
dot = graphviz.Digraph(comment='Proposed Cross-Node Expert Parallelism (64 GPUs, 1 Expert per GPU)', 
                      graph_attr={'rankdir': 'TB', 'bgcolor': 'white', 'ranksep': '1.5', 'nodesep': '0.8'})

# Define node styles
input_style = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightblue', 'color': 'black'}
compute_style = {'shape': 'rectangle', 'style': 'filled', 'fillcolor': 'lightgreen', 'color': 'black'}
comm_style = {'shape': 'ellipse', 'style': 'dashed', 'fillcolor': 'yellow', 'color': 'black'}
route_style = {'shape': 'parallelogram', 'style': 'filled', 'fillcolor': 'lightcoral', 'color': 'black'}
agg_style = {'shape': 'parallelogram', 'style': 'filled', 'fillcolor': 'lightyellow', 'color': 'black'}
output_style = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightblue', 'color': 'black'}

# Input tokens
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='Input Layer', style='dashed')
    c.node('input_tokens', 'Input Tokens\n(B=1024, H=8192)', **input_style)

# Create clusters for each layer
for layer in range(1, 5):
    layer_name = f'layer{layer}'
    
    with dot.subgraph(name=f'cluster_layer{layer}') as c:
        c.attr(label=f'Layer {layer} (64 GPUs)', style='solid')
        
        # MHA computation (duplicated across all GPUs for this layer)
        c.node(f'mha_{layer}', f'MHA Layer {layer}\nAll GPUs\n(B=1024, H=8192)', **compute_style)
        
        # Residual add after MHA
        c.node(f'residual1_{layer}', f'Residual Add\nLayer {layer}\nAll GPUs\n(B=1024, H=8192)', **compute_style)
        
        # Global gate for expert selection
        c.node(f'gate_{layer}', f'Global Gate\nLayer {layer}\nAll GPUs\n(B=1024, E=64)', **route_style)
        
        # Expert distribution across 64 GPUs
        for expert_id in range(1, 65):
            gpu_id = expert_id  # GPU 1-64, one expert per GPU
            expert_name = f'expert_{layer}_gpu{gpu_id}'
            c.node(expert_name, f'Expert {expert_id}\nLayer {layer}\nGPU {gpu_id}\n(B=?, H=32768)', **compute_style)
        
        # Expert aggregation nodes (one per GPU)
        for gpu_id in range(1, 65):
            agg_name = f'agg_{layer}_gpu{gpu_id}'
            c.node(agg_name, f'Expert Agg\nLayer {layer}\nGPU {gpu_id}\n(B=?, H=8192)', **agg_style)
        
        # Global aggregation
        c.node(f'global_agg_{layer}', f'Global Aggregation\nLayer {layer}\nAll GPUs\n(B=1024, H=8192)', **agg_style)
        
        # Final residual add
        c.node(f'residual2_{layer}', f'Final Residual\nLayer {layer}\nAll GPUs\n(B=1024, H=8192)', **compute_style)

# Output
with dot.subgraph(name='cluster_output') as c:
    c.attr(label='Output Layer', style='dashed')
    c.node('output_tokens', 'Output Tokens\n(B=1024, H=8192)', **output_style)

# Communication nodes for cross-node token routing
for layer in range(1, 5):
    for gpu_id in range(1, 65):
        comm_name = f'comm_{layer}_gpu{gpu_id}'
        dot.node(comm_name, f'Token Comm\nLayer {layer}\nGPU {gpu_id}\n(B=?, H=8192)', **comm_style)

# Connect the DAG for each layer
for layer in range(1, 5):
    prev_layer = layer - 1
    
    # Input connections
    if layer == 1:
        dot.edge('input_tokens', 'mha_1')
    else:
        dot.edge(f'residual2_{prev_layer}', f'mha_{layer}')
    
    # MHA to residual
    dot.edge(f'mha_{layer}', f'residual1_{layer}')
    
    # Input to residual (for first layer) or previous output to residual
    if layer == 1:
        dot.edge('input_tokens', f'residual1_{layer}')
    else:
        dot.edge(f'residual2_{prev_layer}', f'residual1_{layer}')
    
    # Residual to gate
    dot.edge(f'residual1_{layer}', f'gate_{layer}')
    
    # Gate to communication (routing tokens to experts)
    for gpu_id in range(1, 65):
        dot.edge(f'gate_{layer}', f'comm_{layer}_gpu{gpu_id}', style='dashed')
        dot.edge(f'comm_{layer}_gpu{gpu_id}', f'expert_{layer}_gpu{gpu_id}')
    
    # Expert to local aggregation
    for gpu_id in range(1, 65):
        dot.edge(f'expert_{layer}_gpu{gpu_id}', f'agg_{layer}_gpu{gpu_id}')
    
    # Local aggregations to global aggregation
    for gpu_id in range(1, 65):
        dot.edge(f'agg_{layer}_gpu{gpu_id}', f'global_agg_{layer}')
    
    # Global aggregation to final residual
    dot.edge(f'global_agg_{layer}', f'residual2_{layer}')
    dot.edge(f'residual1_{layer}', f'residual2_{layer}')  # Residual connection

# Connect final layer to output
dot.edge('residual2_4', 'output_tokens')

# Save the graph
dot.format = 'svg'
dot.render('/home/wzc/data/papers/submission/proposed_moe_deployment', cleanup=True)

print("Proposed cross-node expert parallelism DAG saved to /home/wzc/data/papers/submission/proposed_moe_deployment.svg")