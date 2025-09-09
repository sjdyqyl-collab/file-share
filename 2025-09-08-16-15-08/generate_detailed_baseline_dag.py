#!/usr/bin/env python3

import graphviz

def create_detailed_baseline_dag():
    """Create detailed baseline DAG showing tensor parallelism and expert placement"""
    dot = graphviz.Digraph('detailed_baseline_moe_dag', format='svg')
    dot.attr(rankdir='TB', size='40,30')
    
    # Input specifications
    batch_size = 1024
    seq_len = 10000
    hidden_dim = 8192
    ffn_hidden = 32768
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', f'Input\\n[Batch: {batch_size}×{seq_len}, Dim: {hidden_dim}]', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Process all 4 layers
    for layer_id in range(4):
        stage = layer_id % 2
        gpu_base = 0 if stage == 0 else 8
        
        layer_cluster = f'cluster_layer_{layer_id}'
        with dot.subgraph(name=layer_cluster) as layer:
            layer.attr(label=f'MoE Layer {layer_id} (Pipeline Stage {stage})', 
                      style='dashed', color='blue')
            
            # Gating network
            gate_node = f'gate_{layer_id}'
            layer.node(gate_node, 
                      f'Gating\\nLayer {layer_id}\\n[Input: {hidden_dim}, Output: 16 experts]\\nGPU {gpu_base}',
                      shape='diamond', fillcolor='yellow')
            
            # Expert placement (4 experts per GPU)
            for gpu_id in range(8):
                actual_gpu = gpu_base + gpu_id
                gpu_cluster = f'cluster_gpu_{actual_gpu}_{layer_id}'
                
                with dot.subgraph(name=gpu_cluster) as gpu:
                    gpu.attr(label=f'GPU {actual_gpu}\\n(4 experts)', style='dotted')
                    
                    for expert_idx in range(4):
                        expert_num = layer_id * 16 + gpu_id * 4 + expert_idx
                        expert_node = f'expert_{layer_id}_{expert_num}'
                        
                        # Expert MLP with tensor parallelism
                        gpu.node(f'{expert_node}_linear1', 
                                f'Expert {expert_num}\\nLinear 1\\n[{hidden_dim}→{ffn_hidden}]\\nGPU {actual_gpu}',
                                shape='rectangle', fillcolor='lightcoral')
                        gpu.node(f'{expert_node}_gelu', 
                                f'Expert {expert_num}\\nGELU\\n[{ffn_hidden}]\\nGPU {actual_gpu}',
                                shape='ellipse', fillcolor='lightblue')
                        gpu.node(f'{expert_node}_linear2', 
                                f'Expert {expert_num}\\nLinear 2\\n[{ffn_hidden}→{hidden_dim}]\\nGPU {actual_gpu}',
                                shape='rectangle', fillcolor='lightcoral')
                        
                        # Connect expert components
                        gpu.edge(f'{expert_node}_linear1', f'{expert_node}_gelu')
                        gpu.edge(f'{expert_node}_gelu', f'{expert_node}_linear2')
            
            # Expert aggregation
            layer.node(f'expert_agg_{layer_id}', 
                      f'Expert Aggregation\\nLayer {layer_id}\\n[Combine 16 experts]',
                      shape='parallelogram', fillcolor='purple')
            
            # Layer norm
            layer.node(f'layer_norm_{layer_id}', 
                      f'Layer Norm\\nLayer {layer_id}\\n[{hidden_dim}]',
                      shape='ellipse', fillcolor='lightgreen')
    
    # Communication nodes
    dot.node('tp_allreduce', 'Tensor Parallel\\nAll-Reduce\\n[8 GPUs]', 
             shape='parallelogram', fillcolor='orange')
    dot.node('pp_send', 'Pipeline Send\\n[Stage 0→1]', 
             shape='parallelogram', fillcolor='purple')
    dot.node('pp_recv', 'Pipeline Receive\\n[Stage 1←0]', 
             shape='parallelogram', fillcolor='purple')
    
    # Output node
    dot.node('output', f'Output\\n[Batch: {batch_size}×{seq_len}, Dim: {hidden_dim}]', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Add edges
    # Input to first layer
    dot.edge('input', 'gate_0')
    
    # Layer 0: Gating to experts
    for expert_num in range(16):
        gpu_id = expert_num // 4
        dot.edge('gate_0', f'expert_0_{expert_num}_linear1', 
                style='dashed', label=f'route to GPU {gpu_id}')
    
    # Expert outputs to aggregation
    for expert_num in range(16):
        dot.edge(f'expert_0_{expert_num}_linear2', 'expert_agg_0')
    
    # Layer 0 to Layer 1
    dot.edge('expert_agg_0', 'layer_norm_0')
    dot.edge('layer_norm_0', 'gate_1')
    
    # Layer 1: Similar connections
    for expert_num in range(16, 32):
        gpu_id = (expert_num - 16) // 4 + 8  # Stage 1 GPUs
        dot.edge('gate_1', f'expert_1_{expert_num}_linear1', 
                style='dashed', label=f'route to GPU {gpu_id}')
    
    for expert_num in range(16, 32):
        dot.edge(f'expert_1_{expert_num}_linear2', 'expert_agg_1')
    
    dot.edge('expert_agg_1', 'layer_norm_1')
    dot.edge('layer_norm_1', 'gate_2')
    
    # Layer 2: Back to stage 0
    for expert_num in range(32, 48):
        gpu_id = (expert_num - 32) // 4
        dot.edge('gate_2', f'expert_2_{expert_num}_linear1', 
                style='dashed', label=f'route to GPU {gpu_id}')
    
    for expert_num in range(32, 48):
        dot.edge(f'expert_2_{expert_num}_linear2', 'expert_agg_2')
    
    dot.edge('expert_agg_2', 'layer_norm_2')
    dot.edge('layer_norm_2', 'gate_3')
    
    # Layer 3: Stage 1
    for expert_num in range(48, 64):
        gpu_id = (expert_num - 48) // 4 + 8
        dot.edge('gate_3', f'expert_3_{expert_num}_linear1', 
                style='dashed', label=f'route to GPU {gpu_id}')
    
    for expert_num in range(48, 64):
        dot.edge(f'expert_3_{expert_num}_linear2', 'expert_agg_3')
    
    dot.edge('expert_agg_3', 'layer_norm_3')
    dot.edge('layer_norm_3', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_detailed_baseline_dag()
    dag.render('/home/wzc/data/file-share/2025-09-08-16-15-08/detailed_baseline_moe_dag', format='svg', cleanup=False)
    
    # Save DOT file
    with open('/home/wzc/data/file-share/2025-09-08-16-15-08/detailed_baseline_moe_dag.dot', 'w') as f:
        f.write(dag.source)
    
    print("Detailed baseline DAG generated successfully!")
    print(f"SVG: /home/wzc/data/file-share/2025-09-08-16-15-08/detailed_baseline_moe_dag.svg")
    print(f"DOT: /home/wzc/data/file-share/2025-09-08-16-15-08/detailed_baseline_moe_dag.dot")