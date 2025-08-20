import graphviz

# Create a new directed graph for baseline deployment (TP=8, PP=2, 16 GPUs)
dot = graphviz.Digraph(comment='Baseline MoE Deployment (TP=8, PP=2, 16 GPUs)', 
                      graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})

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

# Pipeline Stage 1 (Layers 1-2) - 8 GPUs
for stage in [1, 2]:
    with dot.subgraph(name=f'cluster_stage{stage}') as c:
        c.attr(label=f'Pipeline Stage {stage} (GPUs 1-8)', style='solid')
        
        for layer in [1, 2] if stage == 1 else [3, 4]:
            layer_name = f'layer{layer}'
            
            # MHA for this layer
            for gpu_id in range(1, 9):
                gpu_name = f'gpu{gpu_id}_stage{stage}'
                
                # MHA computation (TP=8)
                mha_name = f'mha_{layer}_gpu{gpu_id}'
                c.node(mha_name, f'MHA Layer {layer}\nGPU {gpu_id}\n(B=1024, H=1024)', 
                      **compute_style)
                
                # Residual add after MHA
                residual1_name = f'residual1_{layer}_gpu{gpu_id}'
                c.node(residual1_name, f'Residual Add\nGPU {gpu_id}\n(B=1024, H=8192)', 
                      **compute_style)
                
                # MoE experts (4 experts per GPU)
                for expert_id in range(1, 5):
                    expert_name = f'expert_{layer}_gpu{gpu_id}_exp{expert_id}'
                    c.node(expert_name, f'Expert {expert_id}\nLayer {layer}\nGPU {gpu_id}\n(B=?, H=32768)', 
                          **compute_style)
                
                # Gate for expert selection
                gate_name = f'gate_{layer}_gpu{gpu_id}'
                c.node(gate_name, f'Gate\nLayer {layer}\nGPU {gpu_id}\n(B=1024, E=64)', 
                      **route_style)
                
                # Expert aggregation
                agg_name = f'agg_{layer}_gpu{gpu_id}'
                c.node(agg_name, f'Expert Agg\nLayer {layer}\nGPU {gpu_id}\n(B=1024, H=8192)', 
                      **agg_style)
                
                # Final residual add
                residual2_name = f'residual2_{layer}_gpu{gpu_id}'
                c.node(residual2_name, f'Final Residual\nLayer {layer}\nGPU {gpu_id}\n(B=1024, H=8192)', 
                      **compute_style)

# Pipeline communication between stages
for gpu_id in range(1, 9):
    # Stage 1 to Stage 2 communication
    comm_name = f'comm_stage1_to_stage2_gpu{gpu_id}'
    dot.node(comm_name, f'Pipeline Comm\nGPU {gpu_id} Stage1→2\n(B=1024, H=8192)', 
            **comm_style)

# Output
with dot.subgraph(name='cluster_output') as c:
    c.attr(label='Output Layer', style='dashed')
    c.node('output_tokens', 'Output Tokens\n(B=1024, H=8192)', **output_style)

# Connect the DAG
# Input to Layer 1
for gpu_id in range(1, 9):
    dot.edge('input_tokens', f'mha_1_gpu{gpu_id}')
    dot.edge(f'mha_1_gpu{gpu_id}', f'residual1_1_gpu{gpu_id}')
    dot.edge('input_tokens', f'residual1_1_gpu{gpu_id}')  # Residual connection
    dot.edge(f'residual1_1_gpu{gpu_id}', f'gate_1_gpu{gpu_id}')
    
    # Gate to experts
    for expert_id in range(1, 5):
        dot.edge(f'gate_1_gpu{gpu_id}', f'expert_1_gpu{gpu_id}_exp{expert_id}', style='dashed')
    
    # Experts to aggregation
    for expert_id in range(1, 5):
        dot.edge(f'expert_1_gpu{gpu_id}_exp{expert_id}', f'agg_1_gpu{gpu_id}')
    
    dot.edge(f'agg_1_gpu{gpu_id}', f'residual2_1_gpu{gpu_id}')
    dot.edge(f'residual1_1_gpu{gpu_id}', f'residual2_1_gpu{gpu_id}')  # Residual connection

# Layer 1 to Layer 2
for gpu_id in range(1, 9):
    dot.edge(f'residual2_1_gpu{gpu_id}', f'mha_2_gpu{gpu_id}')
    dot.edge(f'mha_2_gpu{gpu_id}', f'residual1_2_gpu{gpu_id}')
    dot.edge(f'residual2_1_gpu{gpu_id}', f'residual1_2_gpu{gpu_id}')  # Residual connection
    dot.edge(f'residual1_2_gpu{gpu_id}', f'gate_2_gpu{gpu_id}')
    
    # Gate to experts
    for expert_id in range(1, 5):
        dot.edge(f'gate_2_gpu{gpu_id}', f'expert_2_gpu{gpu_id}_exp{expert_id}', style='dashed')
    
    # Experts to aggregation
    for expert_id in range(1, 5):
        dot.edge(f'expert_2_gpu{gpu_id}_exp{expert_id}', f'agg_2_gpu{gpu_id}')
    
    dot.edge(f'agg_2_gpu{gpu_id}', f'residual2_2_gpu{gpu_id}')
    dot.edge(f'residual1_2_gpu{gpu_id}', f'residual2_2_gpu{gpu_id}')  # Residual connection
    
    # Pipeline communication
    dot.edge(f'residual2_2_gpu{gpu_id}', f'comm_stage1_to_stage2_gpu{gpu_id}')

# Layer 3 and 4 (Stage 2)
for gpu_id in range(1, 9):
    # Layer 3
    dot.edge(f'comm_stage1_to_stage2_gpu{gpu_id}', f'mha_3_gpu{gpu_id}')
    dot.edge(f'mha_3_gpu{gpu_id}', f'residual1_3_gpu{gpu_id}')
    dot.edge(f'comm_stage1_to_stage2_gpu{gpu_id}', f'residual1_3_gpu{gpu_id}')  # Residual connection
    dot.edge(f'residual1_3_gpu{gpu_id}', f'gate_3_gpu{gpu_id}')
    
    for expert_id in range(1, 5):
        dot.edge(f'gate_3_gpu{gpu_id}', f'expert_3_gpu{gpu_id}_exp{expert_id}', style='dashed')
    
    for expert_id in range(1, 5):
        dot.edge(f'expert_3_gpu{gpu_id}_exp{expert_id}', f'agg_3_gpu{gpu_id}')
    
    dot.edge(f'agg_3_gpu{gpu_id}', f'residual2_3_gpu{gpu_id}')
    dot.edge(f'residual1_3_gpu{gpu_id}', f'residual2_3_gpu{gpu_id}')  # Residual connection
    
    # Layer 4
    dot.edge(f'residual2_3_gpu{gpu_id}', f'mha_4_gpu{gpu_id}')
    dot.edge(f'mha_4_gpu{gpu_id}', f'residual1_4_gpu{gpu_id}')
    dot.edge(f'residual2_3_gpu{gpu_id}', f'residual1_4_gpu{gpu_id}')  # Residual connection
    dot.edge(f'residual1_4_gpu{gpu_id}', f'gate_4_gpu{gpu_id}')
    
    for expert_id in range(1, 5):
        dot.edge(f'gate_4_gpu{gpu_id}', f'expert_4_gpu{gpu_id}_exp{expert_id}', style='dashed')
    
    for expert_id in range(1, 5):
        dot.edge(f'expert_4_gpu{gpu_id}_exp{expert_id}', f'agg_4_gpu{gpu_id}')
    
    dot.edge(f'agg_4_gpu{gpu_id}', f'residual2_4_gpu{gpu_id}')
    dot.edge(f'residual1_4_gpu{gpu_id}', f'residual2_4_gpu{gpu_id}')  # Residual connection
    
    # Connect to output
    dot.edge(f'residual2_4_gpu{gpu_id}', 'output_tokens')

# Save the graph
dot.format = 'svg'
dot.render('/home/wzc/data/papers/submission/baseline_moe_deployment', cleanup=True)

print("Baseline deployment DAG saved to /home/wzc/data/papers/submission/baseline_moe_deployment.svg")