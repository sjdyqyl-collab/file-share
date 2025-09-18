#!/usr/bin/env python3

import graphviz

def create_proposed_dag():
    """Create proposed DAG with 64 GPUs, 1 expert/GPU, EP=16"""
    dot = graphviz.Digraph('proposed_moe_dag', format='svg')
    dot.attr(rankdir='TB', size='40,30')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input\\n[Batch: 1024×10000, Dim: 8192]', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Process all 4 layers with 16 experts each
    for layer_id in range(4):
        layer_cluster = f'cluster_layer_{layer_id}'
        with dot.subgraph(name=layer_cluster) as layer:
            layer.attr(label=f'MoE Layer {layer_id} (Pipeline Stage {layer_id})', 
                      style='dashed', color='blue')
            
            # Gating network for this layer
            gate_node = f'gate_{layer_id}'
            layer.node(gate_node, 
                      f'Gating Network Layer {layer_id}\\n[Input: 8192, Output: 16 experts]\\nGPU {layer_id*16}',
                      shape='diamond', fillcolor='yellow')
            
            # Create 16 experts, one per GPU
            for expert_idx in range(16):
                expert_id = layer_id * 16 + expert_idx
                gpu_id = expert_id
                
                expert_cluster = f'cluster_expert_{expert_id}'
                with layer.subgraph(name=expert_cluster) as expert:
                    expert.attr(label=f'Expert {expert_id}\\nGPU {gpu_id}', 
                               style='rounded', color='red')
                    
                    # Expert MLP structure
                    expert.node(f'expert_{expert_id}_linear1', 
                               f'Linear 1\\n[8192→32768]\\nGPU {gpu_id}',
                               shape='rectangle', fillcolor='lightcoral')
                    expert.node(f'expert_{expert_id}_gelu', 
                               f'GELU\\n[32768]\\nGPU {gpu_id}',
                               shape='ellipse', fillcolor='lightblue')
                    expert.node(f'expert_{expert_id}_linear2', 
                               f'Linear 2\\n[32768→8192]\\nGPU {gpu_id}',
                               shape='rectangle', fillcolor='lightcoral')
                    
                    # Connect expert components
                    expert.edge(f'expert_{expert_id}_linear1', f'expert_{expert_id}_gelu')
                    expert.edge(f'expert_{expert_id}_gelu', f'expert_{expert_id}_linear2')
            
            # Token routing and aggregation
            route_node = f'route_{layer_id}'
            layer.node(route_node, 
                      f'Token Router\\nLayer {layer_id}\\n[Cross-node routing]',
                      shape='parallelogram', fillcolor='orange')
            
            agg_node = f'aggregate_{layer_id}'
            layer.node(agg_node, 
                      f'Token Aggregator\\nLayer {layer_id}\\n[Combine expert outputs]',
                      shape='parallelogram', fillcolor='purple')
    
    # Add communication nodes for cross-node routing
    for layer_id in range(4):
        for expert_idx in range(16):
            expert_id = layer_id * 16 + expert_idx
            comm_node = f'comm_{expert_id}'
            dot.node(comm_node, 
                    f'Token Comm\\nExpert {expert_id}\\n[GPU {expert_id}]',
                    shape='parallelogram', fillcolor='gold')
    
    # Output node
    dot.node('output', 'Output\\n[Batch: 1024×10000, Dim: 8192]', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Add edges for data flow
    # Input to first gating
    dot.edge('input', 'gate_0')
    
    # Layer 0 routing
    for expert_idx in range(16):
        expert_id = expert_idx
        dot.edge('gate_0', f'route_0', style='dashed', label='select')
        dot.edge('route_0', f'expert_{expert_id}_linear1', 
                label=f'to GPU {expert_id}')
        dot.edge(f'expert_{expert_id}_linear2', f'aggregate_0')
    
    # Connect layers
    dot.edge('aggregate_0', 'gate_1')
    
    # Layer 1 routing
    for expert_idx in range(16):
        expert_id = 16 + expert_idx
        dot.edge('gate_1', f'route_1', style='dashed')
        dot.edge('route_1', f'expert_{expert_id}_linear1')
        dot.edge(f'expert_{expert_id}_linear2', f'aggregate_1')
    
    dot.edge('aggregate_1', 'gate_2')
    
    # Layer 2 routing
    for expert_idx in range(16):
        expert_id = 32 + expert_idx
        dot.edge('gate_2', f'route_2', style='dashed')
        dot.edge('route_2', f'expert_{expert_id}_linear1')
        dot.edge(f'expert_{expert_id}_linear2', f'aggregate_2')
    
    dot.edge('aggregate_2', 'gate_3')
    
    # Layer 3 routing
    for expert_idx in range(16):
        expert_id = 48 + expert_idx
        dot.edge('gate_3', f'route_3', style='dashed')
        dot.edge('route_3', f'expert_{expert_id}_linear1')
        dot.edge(f'expert_{expert_id}_linear2', f'aggregate_3')
    
    # Final output
    dot.edge('aggregate_3', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_proposed_dag()
    dag.render('/home/wzc/data/file-share/2025-09-08-16-15-08/proposed_moe_dag', format='svg', cleanup=False)
    
    # Save DOT file
    with open('/home/wzc/data/file-share/2025-09-08-16-15-08/proposed_moe_dag.dot', 'w') as f:
        f.write(dag.source)
    
    print("Proposed DAG generated successfully!")
    print(f"SVG: /home/wzc/data/file-share/2025-09-08-16-15-08/proposed_moe_dag.svg")
    print(f"DOT: /home/wzc/data/file-share/2025-09-08-16-15-08/proposed_moe_dag.dot")