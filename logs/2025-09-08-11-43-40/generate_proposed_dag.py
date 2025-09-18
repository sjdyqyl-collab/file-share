#!/usr/bin/env python3
"""
Generate proposed cross-node expert parallelism DAG with:
- 64 GPUs total
- 1 expert per GPU
- EP=64
- 4 layers total
"""

import graphviz

def create_proposed_dag():
    dot = graphviz.Digraph('proposed_moe_deployment', 
                          comment='Proposed Cross-Node Expert Parallelism DAG',
                          format='svg')
    
    # Set graph attributes
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input Tokens\n[1024, 10000, 8192]', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Create nodes for each layer and GPU
    for layer in range(4):
        with dot.subgraph(name=f'cluster_layer{layer}') as layer_cluster:
            layer_cluster.attr(label=f'Layer {layer}\n16 Experts across 16 GPUs', style='rounded')
            
            # Attention layer - replicated per layer
            attn_name = f'layer{layer}_attention'
            layer_cluster.node(attn_name, 
                             f'Multi-Head Attention\n[1024, 10000, 8192]\n16 heads × 512 dim\nReplicated across layer',
                             fillcolor='lightcoral')
            
            # Gating network
            gate_name = f'layer{layer}_gating'
            layer_cluster.node(gate_name, 
                             f'Top-K Gating\nSelect 2 experts\nfrom 16 total\nLayer {layer}',
                             shape='diamond', fillcolor='orange')
            
            # Token router
            router_name = f'layer{layer}_router'
            layer_cluster.node(router_name, 
                             f'Async Token Router\nRoute tokens to experts\nLayer {layer}',
                             shape='parallelogram', fillcolor='lightpink')
            
            # Expert MLPs - one per GPU
            for expert_id in range(16):
                gpu_id = layer * 16 + expert_id
                expert_name = f'layer{layer}_expert{expert_id}_gpu{gpu_id}'
                layer_cluster.node(expert_name, 
                                 f'Expert {layer*16+expert_id}\n[1024, 10000, 8192]→[1024, 10000, 32768]→[1024, 10000, 8192]\nGPU {gpu_id} (Node {gpu_id//4})',
                                 fillcolor='lightsteelblue')
            
            # Expert aggregator
            aggregator_name = f'layer{layer}_aggregator'
            layer_cluster.node(aggregator_name, 
                             f'Expert Output\nAggregator\nLayer {layer}',
                             shape='parallelogram', fillcolor='lightgreen')
    
    # Communication nodes for cross-node transfers
    dot.attr('node', shape='parallelogram', fillcolor='yellow')
    
    # Inter-node communication nodes
    for layer in range(3):  # Between layers
        for node in range(16):  # 16 nodes total
            comm_name = f'inter_node_comm_layer{layer}_to_{layer+1}_node{node}'
            dot.node(comm_name, 
                    f'Inter-Node Comm\nLayer {layer}→{layer+1}\nNode {node}\n400Gbps InfiniBand',
                    fillcolor='yellow')
    
    # Async communication overlap nodes
    dot.attr('node', shape='parallelogram', fillcolor='lightcyan')
    for layer in range(4):
        overlap_name = f'comm_compute_overlap_layer{layer}'
        dot.node(overlap_name, 
                f'Compute-Communication\nOverlap\nLayer {layer}',
                fillcolor='lightcyan')
    
    # Output node
    dot.node('output', 'Output Tokens\n[1024, 10000, 8192]', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Connect the flow
    # Input to Layer 0
    dot.edge('input', 'layer0_attention')
    dot.edge('layer0_attention', 'layer0_gating')
    dot.edge('layer0_gating', 'layer0_router', style='dashed', label='top-2 selection')
    
    # Layer 0 expert routing and computation
    for expert_id in range(16):
        gpu_id = expert_id
        expert_name = f'layer0_expert{expert_id}_gpu{gpu_id}'
        dot.edge('layer0_router', expert_name, 
                label=f'tokens routed\nto expert {expert_id}')
        dot.edge(expert_name, 'layer0_aggregator')
    
    # Communication overlap for layer 0
    dot.edge('layer0_aggregator', 'comm_compute_overlap_layer0')
    
    # Connect to Layer 1 with inter-node communication
    for node in range(16):
        comm_name = f'inter_node_comm_layer0_to_1_node{node}'
        dot.edge('comm_compute_overlap_layer0', comm_name)
        dot.edge(comm_name, 'layer1_attention')
    
    # Layer 1
    dot.edge('layer1_attention', 'layer1_gating')
    dot.edge('layer1_gating', 'layer1_router', style='dashed', label='top-2 selection')
    
    for expert_id in range(16):
        gpu_id = 16 + expert_id
        expert_name = f'layer1_expert{expert_id}_gpu{gpu_id}'
        dot.edge('layer1_router', expert_name, 
                label=f'tokens routed\nto expert {16+expert_id}')
        dot.edge(expert_name, 'layer1_aggregator')
    
    dot.edge('layer1_aggregator', 'comm_compute_overlap_layer1')
    
    # Connect to Layer 2
    for node in range(16):
        comm_name = f'inter_node_comm_layer1_to_2_node{node}'
        dot.edge('comm_compute_overlap_layer1', comm_name)
        dot.edge(comm_name, 'layer2_attention')
    
    # Layer 2
    dot.edge('layer2_attention', 'layer2_gating')
    dot.edge('layer2_gating', 'layer2_router', style='dashed', label='top-2 selection')
    
    for expert_id in range(16):
        gpu_id = 32 + expert_id
        expert_name = f'layer2_expert{expert_id}_gpu{gpu_id}'
        dot.edge('layer2_router', expert_name, 
                label=f'tokens routed\nto expert {32+expert_id}')
        dot.edge(expert_name, 'layer2_aggregator')
    
    dot.edge('layer2_aggregator', 'comm_compute_overlap_layer2')
    
    # Connect to Layer 3
    for node in range(16):
        comm_name = f'inter_node_comm_layer2_to_3_node{node}'
        dot.edge('comm_compute_overlap_layer2', comm_name)
        dot.edge(comm_name, 'layer3_attention')
    
    # Layer 3
    dot.edge('layer3_attention', 'layer3_gating')
    dot.edge('layer3_gating', 'layer3_router', style='dashed', label='top-2 selection')
    
    for expert_id in range(16):
        gpu_id = 48 + expert_id
        expert_name = f'layer3_expert{expert_id}_gpu{gpu_id}'
        dot.edge('layer3_router', expert_name, 
                label=f'tokens routed\nto expert {48+expert_id}')
        dot.edge(expert_name, 'layer3_aggregator')
    
    dot.edge('layer3_aggregator', 'comm_compute_overlap_layer3')
    dot.edge('comm_compute_overlap_layer3', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_proposed_dag()
    dag.render('/home/wzc/data/file-share/2025-09-08-11-43-40/proposed_moe_deployment', 
               format='svg', cleanup=True)
    
    # Also save as .dot file
    with open('/home/wzc/data/file-share/2025-09-08-11-43-40/proposed_moe_deployment.dot', 'w') as f:
        f.write(dag.source)
    
    print("Proposed DAG generated successfully")