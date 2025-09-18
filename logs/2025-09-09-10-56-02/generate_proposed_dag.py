#!/usr/bin/env python3
"""
Generate proposed DAG for large-scale cross-node expert parallelism with EP=64, 1 expert/GPU
"""

import graphviz

def create_proposed_dag():
    dot = graphviz.Digraph('proposed_moe_dag')
    dot.attr(rankdir='TB', size='30,40')
    dot.attr('node', shape='rectangle', style='filled')
    
    # Global parameters
    batch_size = 1024
    seq_len = 10000
    hidden_dim = 8192
    ffn_hidden = 32768
    num_heads = 16
    head_dim = 512
    vocab_size = 32000
    
    # Color scheme
    input_color = '#E6F3FF'
    compute_color = '#FFE6CC'
    comm_color = '#E6FFE6'
    expert_color = '#FFE6E6'
    routing_color = '#E6E6FF'
    
    # Input layer
    dot.node('input', f'Input\\nBatch={batch_size}, Seq={seq_len}, Hidden={hidden_dim}', 
             shape='ellipse', fillcolor=input_color)
    
    # Process each layer (4 layers total)
    for layer_id in range(4):
        with dot.subgraph(name=f'cluster_layer_{layer_id}') as layer:
            layer.attr(label=f'Layer {layer_id} (64 GPUs, 1 expert/GPU)', style='dashed')
            
            # MHA computation (replicated across all GPUs for each layer)
            for gpu_id in range(64):
                # QKV projection
                dot.node(f'l{layer_id}_qkv_gpu{gpu_id}', 
                         f'QKV Projection\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim*3}',
                         fillcolor=compute_color)
                
                # MHA computation
                dot.node(f'l{layer_id}_mha_gpu{gpu_id}', 
                         f'MHA Computation\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim*3}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         fillcolor=compute_color)
                
                # Output projection
                dot.node(f'l{layer_id}_out_proj_gpu{gpu_id}', 
                         f'Output Projection\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         fillcolor=compute_color)
                
                # First residual
                dot.node(f'l{layer_id}_residual1_gpu{gpu_id}', 
                         f'Residual Add\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         fillcolor=compute_color)
                
                # First layer norm
                dot.node(f'l{layer_id}_layernorm1_gpu{gpu_id}', 
                         f'Layer Norm\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         fillcolor=compute_color)
                
                # Gate computation (determines which tokens go to which expert)
                dot.node(f'l{layer_id}_gate_gpu{gpu_id}', 
                         f'Gate\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x64',
                         shape='parallelogram', fillcolor=routing_color)
                
                # Token routing (communication to target experts)
                for target_gpu in range(64):
                    if target_gpu != gpu_id:
                        dot.node(f'l{layer_id}_route_gpu{gpu_id}_to_{target_gpu}', 
                                 f'Token Route\\nFrom GPU={gpu_id}\\nTo GPU={target_gpu}\\nSize={batch_size}x{seq_len}x{hidden_dim}',
                                 shape='ellipse', fillcolor=comm_color, style='dashed')
                
                # Expert computation (each GPU has exactly one expert)
                expert_id = layer_id * 64 + gpu_id
                dot.node(f'l{layer_id}_expert_gpu{gpu_id}', 
                         f'Expert {expert_id}\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nHidden={batch_size}x{seq_len}x{ffn_hidden}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         fillcolor=expert_color)
                
                # Expert output scaling
                dot.node(f'l{layer_id}_expert_scale_gpu{gpu_id}', 
                         f'Expert Scale\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         fillcolor=compute_color)
                
                # Token aggregation (receive from all GPUs)
                for source_gpu in range(64):
                    if source_gpu != gpu_id:
                        dot.node(f'l{layer_id}_agg_gpu{gpu_id}_from_{source_gpu}', 
                                 f'Token Aggregate\\nFrom GPU={source_gpu}\\nTo GPU={gpu_id}\\nSize={batch_size}x{seq_len}x{hidden_dim}',
                                 shape='ellipse', fillcolor=comm_color, style='dashed')
                
                # Final aggregation
                dot.node(f'l{layer_id}_final_agg_gpu{gpu_id}', 
                         f'Final Aggregation\\nGPU={gpu_id}\\nIn=64x{batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         shape='parallelogram', fillcolor=routing_color)
                
                # Second residual
                dot.node(f'l{layer_id}_residual2_gpu{gpu_id}', 
                         f'Residual Add\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         fillcolor=compute_color)
                
                # Second layer norm
                dot.node(f'l{layer_id}_layernorm2_gpu{gpu_id}', 
                         f'Layer Norm\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         fillcolor=compute_color)
    
    # Output layer
    dot.node('output', f'Output\\nBatch={batch_size}, Seq={seq_len}, Hidden={hidden_dim}',
             shape='ellipse', fillcolor=input_color)
    
    # Connect input to first layer
    for gpu_id in range(64):
        dot.edge('input', f'l0_qkv_gpu{gpu_id}')
    
    # Connect layers
    for layer_id in range(3):  # Connect layer 0->1, 1->2, 2->3
        for gpu_id in range(64):
            dot.edge(f'l{layer_id}_layernorm2_gpu{gpu_id}', f'l{layer_id+1}_qkv_gpu{gpu_id}')
    
    # Connect final layer to output
    for gpu_id in range(64):
        dot.edge(f'l3_layernorm2_gpu{gpu_id}', 'output')
    
    # Save the DAG
    dot.save('/home/wzc/data/file-share/2025-09-09-10-56-02/proposed_moe_dag.dot')
    dot.render('/home/wzc/data/file-share/2025-09-09-10-56-02/proposed_moe_dag', format='svg', cleanup=True)
    
    return '/home/wzc/data/file-share/2025-09-09-10-56-02/proposed_moe_dag.dot'

if __name__ == '__main__':
    create_proposed_dag()