#!/usr/bin/env python3

import graphviz

# Create DAG for MoE Model (16 layers, 8 experts/layer, 16 GPUs, layer-wise deployment)
dot = graphviz.Digraph('MoE_Model_Layerwise_Deployment', 
                      comment='MoE Model Layer-wise Deployment on 16 GPUs')

dot.attr(rankdir='TB', size='25,35')
dot.attr('node', shape='rectangle', style='filled')

# Input node
dot.node('input', 'Input\n[Batch=1024, Seq=2048, Hidden=4096]\nGPU-0', 
         shape='ellipse', fillcolor='lightgreen')

# Layer 1-16, each with 8 experts distributed across GPUs
for layer_idx in range(1, 17):
    gpu_base = (layer_idx - 1)  # Starting GPU for this layer
    
    # Create subgraph for layer
    with dot.subgraph(name=f'cluster_layer_{layer_idx}') as c:
        c.attr(label=f'Layer {layer_idx} (GPU-{gpu_base} to GPU-{gpu_base+7})', 
               style='rounded', fillcolor='lightblue', color='blue')
        
        # Layer norm 1
        c.node(f'ln1_{layer_idx}', f'LayerNorm1\n[1024×2048×4096]\nGPU-{gpu_base}', 
               shape='rectangle', fillcolor='lightyellow')
        
        # Multi-head attention (on GPU-base)
        c.node(f'mha_q_{layer_idx}', f'MHA-Q\n[1024×2048×4096×64×64]\nGPU-{gpu_base}', 
               shape='rectangle', fillcolor='lightcoral')
        c.node(f'mha_k_{layer_idx}', f'MHA-K\n[1024×2048×4096×64×64]\nGPU-{gpu_base}', 
               shape='rectangle', fillcolor='lightcoral')
        c.node(f'mha_v_{layer_idx}', f'MHA-V\n[1024×2048×4096×64×64]\nGPU-{gpu_base}', 
               shape='rectangle', fillcolor='lightcoral')
        c.node(f'mha_attn_{layer_idx}', f'MHA-Attn\n[1024×2048×64×64]\nGPU-{gpu_base}', 
               shape='rectangle', fillcolor='lightcoral')
        c.node(f'mha_out_{layer_idx}', f'MHA-Out\n[1024×2048×4096]\nGPU-{gpu_base}', 
               shape='rectangle', fillcolor='lightcoral')
        
        # Residual connection 1
        c.node(f'residual1_{layer_idx}', f'ResidualAdd1\n[1024×2048×4096]\nGPU-{gpu_base}', 
               shape='parallelogram', fillcolor='lightpink')
        
        # Layer norm 2
        c.node(f'ln2_{layer_idx}', f'LayerNorm2\n[1024×2048×4096]\nGPU-{gpu_base}', 
               shape='rectangle', fillcolor='lightyellow')
        
        # Gate for expert selection
        c.node(f'gate_{layer_idx}', f'Expert Gate\n[1024×2048×4096→8]\nGPU-{gpu_base}', 
               shape='diamond', fillcolor='orange')
        
        # Expert 0-7 (distributed across 8 GPUs)
        experts = []
        for expert_id in range(8):
            expert_gpu = gpu_base + expert_id
            
            # Expert components
            c.node(f'expert{expert_id}_up_{layer_idx}', 
                   f'Expert{expert_id} Up\n[1024×2048×4096→11008]\nGPU-{expert_gpu}', 
                   shape='rectangle', fillcolor='lightgreen')
            c.node(f'expert{expert_id}_act_{layer_idx}', 
                   f'Expert{expert_id} GELU\n[1024×2048×11008]\nGPU-{expert_gpu}', 
                   shape='rectangle', fillcolor='lightgreen')
            c.node(f'expert{expert_id}_down_{layer_idx}', 
                   f'Expert{expert_id} Down\n[1024×2048×11008→4096]\nGPU-{expert_gpu}', 
                   shape='rectangle', fillcolor='lightgreen')
            
            # Expert aggregation
            c.node(f'expert{expert_id}_agg_{layer_idx}', 
                   f'Expert{expert_id} Output\n[1024×2048×4096]\nGPU-{expert_gpu}', 
                   shape='ellipse', fillcolor='lightgray')
            
            experts.append(f'expert{expert_id}_agg_{layer_idx}')
        
        # Expert aggregation across all GPUs
        c.node(f'expert_agg_{layer_idx}', 
               f'Expert Aggregation\n[1024×2048×4096]\nGPU-{gpu_base+7}', 
               shape='parallelogram', fillcolor='gold')
        
        # Final residual connection
        c.node(f'residual2_{layer_idx}', f'ResidualAdd2\n[1024×2048×4096]\nGPU-{gpu_base+7}', 
               shape='parallelogram', fillcolor='lightpink')
        
        # Layer output
        c.node(f'layer_out_{layer_idx}', f'Layer{layer_idx}_Output\n[1024×2048×4096]\nGPU-{gpu_base+7}', 
               shape='ellipse', fillcolor='lightgray')

# Connect layers sequentially with communication
for layer_idx in range(1, 17):
    gpu_base = (layer_idx - 1)
    
    if layer_idx == 1:
        dot.edge('input', f'ln1_{layer_idx}')
    else:
        prev_gpu_base = (layer_idx - 2) * 8 + 7
        curr_gpu_base = (layer_idx - 1)
        dot.edge(f'layer_out_{layer_idx-1}', f'ln1_{layer_idx}', 
                label=f'GPU-{prev_gpu_base}→GPU-{curr_gpu_base}\n[1024×2048×4096]', 
                style='dashed', color='red')
    
    # Internal connections for attention
    dot.edge(f'ln1_{layer_idx}', f'mha_q_{layer_idx}')
    dot.edge(f'ln1_{layer_idx}', f'mha_k_{layer_idx}')
    dot.edge(f'ln1_{layer_idx}', f'mha_v_{layer_idx}')
    dot.edge(f'mha_q_{layer_idx}', f'mha_attn_{layer_idx}')
    dot.edge(f'mha_k_{layer_idx}', f'mha_attn_{layer_idx}')
    dot.edge(f'mha_v_{layer_idx}', f'mha_attn_{layer_idx}')
    dot.edge(f'mha_attn_{layer_idx}', f'mha_out_{layer_idx}')
    
    # Residual connection 1
    if layer_idx == 1:
        dot.edge('input', f'residual1_{layer_idx}')
    else:
        dot.edge(f'layer_out_{layer_idx-1}', f'residual1_{layer_idx}')
    dot.edge(f'mha_out_{layer_idx}', f'residual1_{layer_idx}')
    
    dot.edge(f'residual1_{layer_idx}', f'ln2_{layer_idx}')
    dot.edge(f'ln2_{layer_idx}', f'gate_{layer_idx}')
    
    # Expert routing (dashed lines)
    for expert_id in range(8):
        expert_gpu = gpu_base + expert_id
        dot.edge(f'gate_{layer_idx}', f'expert{expert_id}_up_{layer_idx}', 
                label=f'Route tokens\nGPU-{gpu_base}→GPU-{expert_gpu}', 
                style='dashed', color='blue')
        
        # Expert computation
        dot.edge(f'expert{expert_id}_up_{layer_idx}', f'expert{expert_id}_act_{layer_idx}')
        dot.edge(f'expert{expert_id}_act_{layer_idx}', f'expert{expert_id}_down_{layer_idx}')
        dot.edge(f'expert{expert_id}_down_{layer_idx}', f'expert{expert_id}_agg_{layer_idx}')
        
        # Expert output to aggregation
        dot.edge(f'expert{expert_id}_agg_{layer_idx}', f'expert_agg_{layer_idx}', 
                label=f'GPU-{expert_gpu}→GPU-{gpu_base+7}\n[1024×2048×4096]', 
                style='dashed', color='purple')
    
    # Final connections
    dot.edge(f'expert_agg_{layer_idx}', f'residual2_{layer_idx}')
    dot.edge(f'residual1_{layer_idx}', f'residual2_{layer_idx}')
    dot.edge(f'residual2_{layer_idx}', f'layer_out_{layer_idx}')

# Final output
final_gpu = 15 * 8 + 7  # Last GPU of last layer
dot.node('final_output', 'Final Output\n[1024×2048×4096]\nGPU-127', 
         shape='ellipse', fillcolor='gold')
dot.edge('layer_out_16', 'final_output')

# Save the DAG
output_path = '/home/wzc/data/file-share/submission/moe_model_dag'
dot.render(output_path, format='svg', cleanup=False)
print(f"MoE Model DAG saved to: {output_path}.svg")