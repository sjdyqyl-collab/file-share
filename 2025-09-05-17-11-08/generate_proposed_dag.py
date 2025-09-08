#!/usr/bin/env python3

import graphviz

def create_proposed_dag():
    """
    Create proposed DAG with cross-node expert parallelism
    64 GPUs, 1 expert per GPU, 16 experts per layer × 4 layers = 64 experts
    """
    dot = graphviz.Digraph('proposed_moe_deployment', 
                          comment='Proposed MoE Deployment: Cross-Node Expert Parallelism, 64 GPUs')
    
    # Set graph attributes
    dot.attr(rankdir='TB', size='30,30')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Input
    dot.node('input', 'Input\n[1024, 10000, 8192]', shape='ellipse', fillcolor='lightgreen')
    
    # Process all 4 layers
    for layer in range(4):
        with dot.subgraph(name=f'cluster_layer{layer}') as layer_cluster:
            layer_cluster.attr(label=f'Layer {layer} (16 Experts × 4 GPUs = 64 GPUs)', 
                             style='dashed', color='blue')
            
            # MHA for this layer (distributed across all 64 GPUs)
            for gpu in range(64):
                layer_cluster.node(f'mha{layer}_gpu{gpu}', 
                                 f'MHA-{layer}\n[1024, 10000, 8192/64]\nGPU{gpu}', 
                                 fillcolor='lightcoral')
            
            # Gate for this layer (routing tokens to experts)
            layer_cluster.node(f'gate{layer}', 
                             f'Gate-{layer}\n[1024, 10000, 16]\nAll GPUs', 
                             shape='parallelogram', fillcolor='lightpink')
            
            # 16 experts for this layer (each on separate GPU)
            for expert_id in range(16):
                gpu_start = layer * 16 + expert_id * 4  # 4 GPUs per expert for TP
                layer_cluster.node(f'expert{layer}_{expert_id}', 
                                 f'Expert-{layer}-{expert_id}\n[1024, 10000, 32768]\nGPU{gpu_start}-{gpu_start+3}', 
                                 fillcolor='lightyellow')
            
            # Residual add for this layer
            layer_cluster.node(f'residual{layer}', 
                             f'Residual-{layer}\n[1024, 10000, 8192]\nAll GPUs', 
                             fillcolor='lightgray')
            
            # Communication nodes for token aggregation
            layer_cluster.node(f'agg{layer}', 
                             f'Aggregate-{layer}\n[1024, 10000, 8192]\nAll GPUs', 
                             shape='parallelogram', fillcolor='lightcyan')
    
    # Output
    dot.node('output', 'Output\n[1024, 10000, 8192]', shape='ellipse', fillcolor='lightgreen')
    
    # Connections between layers
    for layer in range(4):
        if layer == 0:
            # Input to first layer
            for gpu in range(64):
                dot.edge('input', f'mha{layer}_gpu{gpu}')
        else:
            # Previous layer to current layer
            for gpu in range(64):
                dot.edge(f'agg{layer-1}', f'mha{layer}_gpu{gpu}')
        
        # Within each layer
        for gpu in range(64):
            dot.edge(f'mha{layer}_gpu{gpu}', f'gate{layer}')
        
        # Gate to experts (dashed lines for selection)
        for expert_id in range(16):
            dot.edge(f'gate{layer}', f'expert{layer}_{expert_id}', style='dashed')
        
        # Experts to aggregation
        for expert_id in range(16):
            dot.edge(f'expert{layer}_{expert_id}', f'agg{layer}')
        
        # Gate to residual (for non-selected tokens)
        dot.edge(f'gate{layer}', f'residual{layer}', style='dashed')
        
        # Aggregation to residual
        dot.edge(f'agg{layer}', f'residual{layer}')
        
        # Residual to next layer (handled by next layer connections)
    
    # Final output
    dot.edge('residual3', 'output')
    
    return dot

def create_detailed_proposed_dag():
    """
    Create a more detailed version showing exact GPU allocations
    """
    dot = graphviz.Digraph('detailed_proposed_moe_deployment', 
                          comment='Detailed Proposed MoE Deployment: 64 GPUs, 1 Expert per GPU')
    
    # Set graph attributes
    dot.attr(rankdir='TB', size='40,40')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Input
    dot.node('input', 'Input\n[1024, 10000, 8192]', shape='ellipse', fillcolor='lightgreen')
    
    # Create nodes for each GPU across all layers
    for layer in range(4):
        layer_start_gpu = layer * 16
        
        with dot.subgraph(name=f'cluster_layer{layer}_detailed') as layer_cluster:
            layer_cluster.attr(label=f'Layer {layer} (GPUs {layer_start_gpu}-{layer_start_gpu+15})', 
                             style='dashed', color='blue')
            
            # MHA distributed across 64 GPUs (16 per layer)
            with layer_cluster.subgraph(name=f'cluster_mha{layer}') as mha_cluster:
                mha_cluster.attr(label='Multi-Head Attention', style='dotted', color='green')
                for gpu in range(16):
                    actual_gpu = layer_start_gpu + gpu
                    mha_cluster.node(f'mha_q{layer}_{gpu}', 
                                   f'MHA-Q-{layer}\n[1024, 10000, 8192/16]\nGPU{actual_gpu}', 
                                   fillcolor='lightcoral')
                    mha_cluster.node(f'mha_k{layer}_{gpu}', 
                                   f'MHA-K-{layer}\n[1024, 10000, 8192/16]\nGPU{actual_gpu}', 
                                   fillcolor='lightcoral')
                    mha_cluster.node(f'mha_v{layer}_{gpu}', 
                                   f'MHA-V-{layer}\n[1024, 10000, 8192/16]\nGPU{actual_gpu}', 
                                   fillcolor='lightcoral')
                    mha_cluster.node(f'mha_out{layer}_{gpu}', 
                                   f'MHA-Out-{layer}\n[1024, 10000, 8192/16]\nGPU{actual_gpu}', 
                                   fillcolor='lightcoral')
            
            # Gate for routing
            layer_cluster.node(f'gate{layer}_detailed', 
                             f'Gate-{layer}\n[1024, 10000, 16]\nAll GPUs', 
                             shape='parallelogram', fillcolor='lightpink')
            
            # 16 experts, each on separate GPU
            with layer_cluster.subgraph(name=f'cluster_experts{layer}') as experts_cluster:
                experts_cluster.attr(label='Experts (16 total)', style='dotted', color='orange')
                for expert_id in range(16):
                    gpu_id = layer_start_gpu + expert_id
                    experts_cluster.node(f'expert{layer}_gpu{expert_id}_fc1', 
                                       f'Expert-{layer}-{expert_id}-FC1\n[1024, 10000, 32768]\nGPU{gpu_id}', 
                                       fillcolor='lightyellow')
                    experts_cluster.node(f'expert{layer}_gpu{expert_id}_gelu', 
                                       f'Expert-{layer}-{expert_id}-GELU\n[1024, 10000, 32768]\nGPU{gpu_id}', 
                                       fillcolor='lightyellow')
                    experts_cluster.node(f'expert{layer}_gpu{expert_id}_fc2', 
                                       f'Expert-{layer}-{expert_id}-FC2\n[1024, 10000, 8192]\nGPU{gpu_id}', 
                                       fillcolor='lightyellow')
            
            # Aggregation and residual
            layer_cluster.node(f'agg{layer}_detailed', 
                             f'Aggregate-{layer}\n[1024, 10000, 8192]\nAll GPUs', 
                             shape='parallelogram', fillcolor='lightcyan')
            layer_cluster.node(f'residual{layer}_detailed', 
                             f'Residual-{layer}\n[1024, 10000, 8192]\nAll GPUs', 
                             fillcolor='lightgray')
    
    # Output
    dot.node('output_detailed', 'Output\n[1024, 10000, 8192]', shape='ellipse', fillcolor='lightgreen')
    
    # Detailed connections
    for layer in range(4):
        layer_start_gpu = layer * 16
        
        if layer == 0:
            # Input to MHA
            for gpu in range(16):
                actual_gpu = layer_start_gpu + gpu
                dot.edge('input', f'mha_q{layer}_{gpu}')
                dot.edge('input', f'mha_k{layer}_{gpu}')
                dot.edge('input', f'mha_v{layer}_{gpu}')
        else:
            # Previous layer to current layer
            for gpu in range(16):
                dot.edge(f'residual{layer-1}_detailed', f'mha_q{layer}_{gpu}')
                dot.edge(f'residual{layer-1}_detailed', f'mha_k{layer}_{gpu}')
                dot.edge(f'residual{layer-1}_detailed', f'mha_v{layer}_{gpu}')
        
        # MHA internal connections
        for gpu in range(16):
            dot.edge(f'mha_q{layer}_{gpu}', f'mha_out{layer}_{gpu}')
            dot.edge(f'mha_k{layer}_{gpu}', f'mha_out{layer}_{gpu}')
            dot.edge(f'mha_v{layer}_{gpu}', f'mha_out{layer}_{gpu}')
            dot.edge(f'mha_out{layer}_{gpu}', f'gate{layer}_detailed')
        
        # Gate to experts
        for expert_id in range(16):
            dot.edge(f'gate{layer}_detailed', f'expert{layer}_gpu{expert_id}_fc1', style='dashed')
        
        # Expert internal connections
        for expert_id in range(16):
            dot.edge(f'expert{layer}_gpu{expert_id}_fc1', f'expert{layer}_gpu{expert_id}_gelu')
            dot.edge(f'expert{layer}_gpu{expert_id}_gelu', f'expert{layer}_gpu{expert_id}_fc2')
            dot.edge(f'expert{layer}_gpu{expert_id}_fc2', f'agg{layer}_detailed')
        
        # Gate to residual (bypass)
        dot.edge(f'gate{layer}_detailed', f'residual{layer}_detailed', style='dashed')
        
        # Aggregation to residual
        dot.edge(f'agg{layer}_detailed', f'residual{layer}_detailed')
    
    # Final output
    dot.edge('residual3_detailed', 'output_detailed')
    
    return dot

if __name__ == '__main__':
    # Generate both versions
    dag1 = create_proposed_dag()
    dag1.render('/home/wzc/data/file-share/2025-09-05-17-11-08/proposed_moe_deployment', format='svg', cleanup=False)
    dag1.save('/home/wzc/data/file-share/2025-09-05-17-11-08/proposed_moe_deployment.dot')
    
    dag2 = create_detailed_proposed_dag()
    dag2.render('/home/wzc/data/file-share/2025-09-05-17-11-08/detailed_proposed_moe_deployment', format='svg', cleanup=False)
    dag2.save('/home/wzc/data/file-share/2025-09-05-17-11-08/detailed_proposed_moe_deployment.dot')
    
    print("Proposed DAGs generated successfully")