#!/usr/bin/env python3

import graphviz

# Create DAG for Dense Model (16 layers, 16 GPUs, layer-wise deployment)
dot = graphviz.Digraph('Dense_Model_Layerwise_Deployment', 
                      comment='Dense Model Layer-wise Deployment on 16 GPUs')

dot.attr(rankdir='TB', size='20,30')
dot.attr('node', shape='rectangle', style='filled')

# Input node
dot.node('input', 'Input\n[Batch=1024, Seq=2048, Hidden=4096]\nGPU-0', 
         shape='ellipse', fillcolor='lightgreen')

# Layer 1-16, each on separate GPU
layers = []
for i in range(1, 17):
    gpu_id = i - 1  # GPU 0-15
    
    # Create subgraph for each GPU
    with dot.subgraph(name=f'cluster_gpu_{gpu_id}') as c:
        c.attr(label=f'GPU-{gpu_id}', style='rounded', fillcolor='lightblue', color='blue')
        
        # Layer norm 1
        c.node(f'ln1_{i}', f'LayerNorm1\n[1024×2048×4096]\nGPU-{gpu_id}', 
               shape='rectangle', fillcolor='lightyellow')
        
        # Multi-head attention
        c.node(f'mha_q_{i}', f'MHA-Q\n[1024×2048×4096×64×64]\nGPU-{gpu_id}', 
               shape='rectangle', fillcolor='lightcoral')
        c.node(f'mha_k_{i}', f'MHA-K\n[1024×2048×4096×64×64]\nGPU-{gpu_id}', 
               shape='rectangle', fillcolor='lightcoral')
        c.node(f'mha_v_{i}', f'MHA-V\n[1024×2048×4096×64×64]\nGPU-{gpu_id}', 
               shape='rectangle', fillcolor='lightcoral')
        c.node(f'mha_attn_{i}', f'MHA-Attn\n[1024×2048×64×64]\nGPU-{gpu_id}', 
               shape='rectangle', fillcolor='lightcoral')
        c.node(f'mha_out_{i}', f'MHA-Out\n[1024×2048×4096]\nGPU-{gpu_id}', 
               shape='rectangle', fillcolor='lightcoral')
        
        # Residual connection 1
        c.node(f'residual1_{i}', f'ResidualAdd1\n[1024×2048×4096]\nGPU-{gpu_id}', 
               shape='parallelogram', fillcolor='lightpink')
        
        # Layer norm 2
        c.node(f'ln2_{i}', f'LayerNorm2\n[1024×2048×4096]\nGPU-{gpu_id}', 
               shape='rectangle', fillcolor='lightyellow')
        
        # FFN
        c.node(f'ffn_up_{i}', f'FFN-Up\n[1024×2048×4096→11008]\nGPU-{gpu_id}', 
               shape='rectangle', fillcolor='lightblue')
        c.node(f'ffn_act_{i}', f'GELU\n[1024×2048×11008]\nGPU-{gpu_id}', 
               shape='rectangle', fillcolor='lightgreen')
        c.node(f'ffn_down_{i}', f'FFN-Down\n[1024×2048×11008→4096]\nGPU-{gpu_id}', 
               shape='rectangle', fillcolor='lightblue')
        
        # Residual connection 2
        c.node(f'residual2_{i}', f'ResidualAdd2\n[1024×2048×4096]\nGPU-{gpu_id}', 
               shape='parallelogram', fillcolor='lightpink')
        
        # Output of layer
        c.node(f'layer_out_{i}', f'Layer{i}_Output\n[1024×2048×4096]\nGPU-{gpu_id}', 
               shape='ellipse', fillcolor='lightgray')

# Connect layers sequentially with communication
for i in range(1, 17):
    if i == 1:
        dot.edge('input', f'ln1_{i}')
    else:
        # Communication between GPUs
        prev_gpu = i - 2
        curr_gpu = i - 1
        dot.edge(f'layer_out_{i-1}', f'ln1_{i}', 
                label=f'GPU-{prev_gpu}→GPU-{curr_gpu}\n[1024×2048×4096]', 
                style='dashed', color='red')
    
    # Internal connections within each layer
    dot.edge(f'ln1_{i}', f'mha_q_{i}')
    dot.edge(f'ln1_{i}', f'mha_k_{i}')
    dot.edge(f'ln1_{i}', f'mha_v_{i}')
    dot.edge(f'mha_q_{i}', f'mha_attn_{i}')
    dot.edge(f'mha_k_{i}', f'mha_attn_{i}')
    dot.edge(f'mha_v_{i}', f'mha_attn_{i}')
    dot.edge(f'mha_attn_{i}', f'mha_out_{i}')
    
    # Residual connection 1 (two inputs)
    if i == 1:
        dot.edge('input', f'residual1_{i}')
    else:
        dot.edge(f'layer_out_{i-1}', f'residual1_{i}')
    dot.edge(f'mha_out_{i}', f'residual1_{i}')
    
    dot.edge(f'residual1_{i}', f'ln2_{i}')
    dot.edge(f'ln2_{i}', f'ffn_up_{i}')
    dot.edge(f'ffn_up_{i}', f'ffn_act_{i}')
    dot.edge(f'ffn_act_{i}', f'ffn_down_{i}')
    
    # Residual connection 2 (two inputs)
    dot.edge(f'residual1_{i}', f'residual2_{i}')
    dot.edge(f'ffn_down_{i}', f'residual2_{i}')
    dot.edge(f'residual2_{i}', f'layer_out_{i}')

# Final output
final_gpu = 15
dot.node('final_output', 'Final Output\n[1024×2048×4096]\nGPU-15', 
         shape='ellipse', fillcolor='gold')
dot.edge('layer_out_16', 'final_output')

# Save the DAG
output_path = '/home/wzc/data/file-share/submission/dense_model_dag'
dot.render(output_path, format='svg', cleanup=False)
print(f"Dense Model DAG saved to: {output_path}.svg")