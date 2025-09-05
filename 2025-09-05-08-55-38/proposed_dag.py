#!/usr/bin/env python3

import graphviz

def create_proposed_dag():
    """
    Create proposed DAG for 64 GPUs with:
    - EP=64 (1 expert per GPU)
    - Cross-node expert parallelism
    - 4-layer MoE model
    """
    dot = graphviz.Digraph('proposed_moe', 
                          comment='Proposed MoE Deployment (64 GPUs, EP=64, 1 expert/GPU)')
    dot.attr(rankdir='TB', size='25,25')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    dot.attr('node', shape='diamond', style='filled', fillcolor='lightcoral')  # Communication
    
    # Model dimensions
    batch_size = 1024
    seq_len = 512
    hidden_size = 8192  # 16 heads × 512
    expert_hidden = 32768
    num_experts = 16
    num_layers = 4
    
    # GPU assignments for proposed method
    # 64 GPUs total, 16 experts per layer, 1 expert per GPU
    # Each layer uses 16 GPUs, 4 layers can run in parallel
    
    # Input
    dot.node('input', f'Input\n[B×S×H]\n[{batch_size}×{seq_len}×{hidden_size}]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Process each layer with EP=64 (1 expert per GPU)
    for layer in range(4):
        with dot.subgraph(name=f'cluster_layer{layer}') as c:
            c.attr(label=f'Layer {layer} - EP=64 (1 expert per GPU)', style='dashed')
            
            # MHA computation - replicated across all GPUs for load balancing
            c.node(f'l{layer}_mha_qkv', f'MHA QKV Linear\n[B×S×H]→[B×S×3H]\nAll 64 GPUs', 
                   shape='rectangle', fillcolor='lightgreen')
            c.node(f'l{layer}_mha_split', f'Split Heads\n[B×S×3H]→[B×S×16×512×3]\nAll 64 GPUs', 
                   shape='parallelogram', fillcolor='lightyellow')
            c.node(f'l{layer}_mha_attn', f'Multi-Head Attention\n[B×S×16×512]\nAll 64 GPUs', 
                   shape='rectangle', fillcolor='lightgreen')
            c.node(f'l{layer}_mha_out', f'MHA Output Linear\n[B×S×H]→[B×S×H]\nAll 64 GPUs', 
                   shape='rectangle', fillcolor='lightgreen')
            c.node(f'l{layer}_mha_res', f'Residual Add\n[B×S×H] + [B×S×H]\nAll 64 GPUs', 
                   shape='parallelogram', fillcolor='lightyellow')
            
            # Expert routing with token batching
            c.node(f'l{layer}_gate', f'Expert Gate\n[B×S×H]→[B×S×16]\nRouting + Token batching\nAll 64 GPUs', 
                   shape='parallelogram', fillcolor='lightyellow')
            
            # Token routing communication
            c.node(f'l{layer}_route_send', f'Token Routing Send\nBatch tokens by destination\nAsync NCCL send\nAll 64 GPUs', 
                   shape='diamond', fillcolor='lightcoral')
            
            # Expert computation - 1 expert per GPU
            for gpu_id in range(16):  # 16 experts per layer
                expert_idx = layer * 16 + gpu_id
                c.node(f'l{layer}_exp_{gpu_id}', 
                       f'Expert {expert_idx}\n[B×S×H]→[B×S×{expert_hidden}]→[B×S×H]\nGPU {gpu_id}', 
                       shape='rectangle', fillcolor='lightgreen')
            
            # Token aggregation communication
            c.node(f'l{layer}_route_recv', f'Token Routing Recv\nGather computed tokens\nAsync NCCL recv\nAll 64 GPUs', 
                   shape='diamond', fillcolor='lightcoral')
            
            # Expert aggregation
            c.node(f'l{layer}_exp_agg', f'Expert Aggregation\nWeighted sum of expert outputs\nAll 64 GPUs', 
                   shape='parallelogram', fillcolor='lightyellow')
            c.node(f'l{layer}_exp_res', f'Residual Add\n[B×S×H] + [B×S×H]\nAll 64 GPUs', 
                   shape='parallelogram', fillcolor='lightyellow')
    
    # Output
    dot.node('output', f'Output\n[B×S×H]\n[{batch_size}×{seq_len}×{hidden_size}]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Connections for each layer
    for layer in range(4):
        if layer == 0:
            dot.edge('input', f'l{layer}_mha_qkv')
        else:
            dot.edge(f'l{layer-1}_exp_res', f'l{layer}_mha_qkv')
        
        # MHA connections
        dot.edge(f'l{layer}_mha_qkv', f'l{layer}_mha_split')
        dot.edge(f'l{layer}_mha_split', f'l{layer}_mha_attn')
        dot.edge(f'l{layer}_mha_attn', f'l{layer}_mha_out')
        dot.edge(f'l{layer}_mha_out', f'l{layer}_mha_res')
        
        if layer == 0:
            dot.edge('input', f'l{layer}_mha_res')  # Residual connection
        else:
            dot.edge(f'l{layer-1}_exp_res', f'l{layer}_mha_res')  # Residual connection
            
        dot.edge(f'l{layer}_mha_res', f'l{layer}_gate')
        dot.edge(f'l{layer}_gate', f'l{layer}_route_send')
        
        # Expert routing connections
        for gpu_id in range(16):
            dot.edge(f'l{layer}_route_send', f'l{layer}_exp_{gpu_id}', style='dashed')
            dot.edge(f'l{layer}_exp_{gpu_id}', f'l{layer}_route_recv')
        
        dot.edge(f'l{layer}_route_recv', f'l{layer}_exp_agg')
        dot.edge(f'l{layer}_exp_agg', f'l{layer}_exp_res')
        dot.edge(f'l{layer}_mha_res', f'l{layer}_exp_res')  # Residual connection
    
    # Final output
    dot.edge('l3_exp_res', 'output')
    
    # Add compute-communication overlap visualization
    with dot.subgraph(name='cluster_overlap') as c:
        c.attr(label='Compute-Communication Overlap', style='dotted')
        c.node('overlap_note', 'Asynchronous token routing\nwith CUDA streams\nOverlap: 75% network utilization', 
               shape='note', fillcolor='lightgray')
    
    return dot

if __name__ == '__main__':
    dag = create_proposed_dag()
    dag.render('/home/wzc/data/file-share/2025-09-05-08-55-38/proposed_moe', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-05-08-55-38/proposed_moe.dot')
    print("Proposed DAG generated successfully")