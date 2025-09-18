#!/usr/bin/env python3
"""
Generate baseline DAG for traditional MoE parallelism with TP=8, PP=2, 4 experts/GPU
"""

import graphviz

def create_baseline_dag():
    dot = graphviz.Digraph('baseline_moe_dag')
    dot.attr(rankdir='TB', size='20,30')
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
    
    # Pipeline stage 0 (GPUs 0-7)
    with dot.subgraph(name='cluster_pipeline_0') as c0:
        c0.attr(label='Pipeline Stage 0 (GPUs 0-7)', style='dashed')
        
        # Layer 0
        with c0.subgraph(name='cluster_layer_0') as l0:
            l0.attr(label='Layer 0', style='dotted')
            
            # MHA across 8 GPUs with tensor parallelism
            for gpu_id in range(8):
                # QKV projection (column parallel)
                qkv_input_dim = hidden_dim // 8 if gpu_id < 7 else hidden_dim - (hidden_dim // 8) * 7
                qkv_output_dim = (hidden_dim * 3) // 8
                
                dot.node(f'l0_qkv_gpu{gpu_id}', 
                         f'QKV Projection\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{qkv_input_dim}\\nOut={batch_size}x{seq_len}x{qkv_output_dim}',
                         fillcolor=compute_color)
                
                # MHA heads (each GPU handles 2 heads)
                heads_per_gpu = 2
                mha_output_dim = (heads_per_gpu * head_dim)
                
                dot.node(f'l0_mha_gpu{gpu_id}', 
                         f'MHA Heads\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{qkv_output_dim}\\nOut={batch_size}x{seq_len}x{mha_output_dim}',
                         fillcolor=compute_color)
                
                # Output projection (row parallel)
                dot.node(f'l0_out_proj_gpu{gpu_id}', 
                         f'Output Projection\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{mha_output_dim}\\nOut={batch_size}x{seq_len}x{qkv_input_dim}',
                         fillcolor=compute_color)
                
                # Residual connections
                dot.node(f'l0_residual_gpu{gpu_id}', 
                         f'Residual Add\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         fillcolor=compute_color)
                
                # Layer norm
                dot.node(f'l0_layernorm_gpu{gpu_id}', 
                         f'Layer Norm\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         fillcolor=compute_color)
                
                # All-reduce operations
                dot.node(f'l0_allreduce1_gpu{gpu_id}', 
                         f'All-Reduce\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         shape='ellipse', fillcolor=comm_color)
                
                # Experts (4 per GPU)
                for expert_id in range(4):
                    expert_num = gpu_id * 4 + expert_id
                    
                    # Gate computation
                    dot.node(f'l0_gate_gpu{gpu_id}_exp{expert_id}', 
                             f'Gate\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x16',
                             shape='parallelogram', fillcolor=routing_color)
                    
                    # Expert MLP
                    dot.node(f'l0_expert_gpu{gpu_id}_exp{expert_id}', 
                             f'Expert {expert_num}\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nHidden={batch_size}x{seq_len}x{ffn_hidden}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                             fillcolor=expert_color)
                    
                    # Expert output scaling
                    dot.node(f'l0_expert_scale_gpu{gpu_id}_exp{expert_id}', 
                             f'Expert Scale\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                             fillcolor=compute_color)
                
                # Expert aggregation
                dot.node(f'l0_expert_agg_gpu{gpu_id}', 
                         f'Expert Aggregation\\nGPU={gpu_id}\\nIn=4x{batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         shape='parallelogram', fillcolor=routing_color)
                
                # Second residual
                dot.node(f'l0_residual2_gpu{gpu_id}', 
                         f'Residual Add\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         fillcolor=compute_color)
                
                # Second layer norm
                dot.node(f'l0_layernorm2_gpu{gpu_id}', 
                         f'Layer Norm\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         fillcolor=compute_color)
                
                # All-reduce operations
                dot.node(f'l0_allreduce2_gpu{gpu_id}', 
                         f'All-Reduce\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         shape='ellipse', fillcolor=comm_color)
        
        # Layer 1 (same structure as layer 0)
        with c0.subgraph(name='cluster_layer_1') as l1:
            l1.attr(label='Layer 1', style='dotted')
            
            for gpu_id in range(8):
                # Similar structure as layer 0, but experts 16-31
                for expert_id in range(4):
                    actual_expert_id = 16 + gpu_id * 4 + expert_id
                    
                    dot.node(f'l1_gate_gpu{gpu_id}_exp{expert_id}', 
                             f'Gate\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x16',
                             shape='parallelogram', fillcolor=routing_color)
                    
                    dot.node(f'l1_expert_gpu{gpu_id}_exp{expert_id}', 
                             f'Expert {actual_expert_id}\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nHidden={batch_size}x{seq_len}x{ffn_hidden}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                             fillcolor=expert_color)
                    
                    dot.node(f'l1_expert_scale_gpu{gpu_id}_exp{expert_id}', 
                             f'Expert Scale\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                             fillcolor=compute_color)
                
                dot.node(f'l1_expert_agg_gpu{gpu_id}', 
                         f'Expert Aggregation\\nGPU={gpu_id}\\nIn=4x{batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         shape='parallelogram', fillcolor=routing_color)
    
    # Pipeline stage 1 (GPUs 8-15)
    with dot.subgraph(name='cluster_pipeline_1') as c1:
        c1.attr(label='Pipeline Stage 1 (GPUs 8-15)', style='dashed')
        
        # Layer 2
        with c1.subgraph(name='cluster_layer_2') as l2:
            l2.attr(label='Layer 2', style='dotted')
            
            for gpu_id in range(8, 16):
                actual_gpu_id = gpu_id - 8
                for expert_id in range(4):
                    actual_expert_id = 32 + actual_gpu_id * 4 + expert_id
                    
                    dot.node(f'l2_gate_gpu{gpu_id}_exp{expert_id}', 
                             f'Gate\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x16',
                             shape='parallelogram', fillcolor=routing_color)
                    
                    dot.node(f'l2_expert_gpu{gpu_id}_exp{expert_id}', 
                             f'Expert {actual_expert_id}\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nHidden={batch_size}x{seq_len}x{ffn_hidden}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                             fillcolor=expert_color)
                    
                    dot.node(f'l2_expert_scale_gpu{gpu_id}_exp{expert_id}', 
                             f'Expert Scale\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                             fillcolor=compute_color)
                
                dot.node(f'l2_expert_agg_gpu{gpu_id}', 
                         f'Expert Aggregation\\nGPU={gpu_id}\\nIn=4x{batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         shape='parallelogram', fillcolor=routing_color)
        
        # Layer 3
        with c1.subgraph(name='cluster_layer_3') as l3:
            l3.attr(label='Layer 3', style='dotted')
            
            for gpu_id in range(8, 16):
                actual_gpu_id = gpu_id - 8
                for expert_id in range(4):
                    actual_expert_id = 48 + actual_gpu_id * 4 + expert_id
                    
                    dot.node(f'l3_gate_gpu{gpu_id}_exp{expert_id}', 
                             f'Gate\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x16',
                             shape='parallelogram', fillcolor=routing_color)
                    
                    dot.node(f'l3_expert_gpu{gpu_id}_exp{expert_id}', 
                             f'Expert {actual_expert_id}\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nHidden={batch_size}x{seq_len}x{ffn_hidden}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                             fillcolor=expert_color)
                    
                    dot.node(f'l3_expert_scale_gpu{gpu_id}_exp{expert_id}', 
                             f'Expert Scale\\nGPU={gpu_id}\\nIn={batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                             fillcolor=compute_color)
                
                dot.node(f'l3_expert_agg_gpu{gpu_id}', 
                         f'Expert Aggregation\\nGPU={gpu_id}\\nIn=4x{batch_size}x{seq_len}x{hidden_dim}\\nOut={batch_size}x{seq_len}x{hidden_dim}',
                         shape='parallelogram', fillcolor=routing_color)
    
    # Pipeline communication
    dot.node('pipeline_send', f'Pipeline Send\\nFrom GPUs 0-7\\nTo GPUs 8-15\\nSize={batch_size}x{seq_len}x{hidden_dim}',
             shape='ellipse', fillcolor=comm_color)
    
    # Output
    dot.node('output', f'Output\\nBatch={batch_size}, Seq={seq_len}, Hidden={hidden_dim}',
             shape='ellipse', fillcolor=input_color)
    
    # Connect input to first layer
    for gpu_id in range(8):
        dot.edge('input', f'l0_qkv_gpu{gpu_id}')
    
    # Connect pipeline stages
    for gpu_id in range(8):
        dot.edge(f'l1_allreduce2_gpu{gpu_id}', 'pipeline_send')
    
    for gpu_id in range(8, 16):
        dot.edge('pipeline_send', f'l2_gate_gpu{gpu_id}_exp0')
        dot.edge(f'l3_allreduce2_gpu{gpu_id}', 'output')
    
    # Save the DAG
    dot.save('/home/wzc/data/file-share/2025-09-09-10-56-02/baseline_moe_dag.dot')
    dot.render('/home/wzc/data/file-share/2025-09-09-10-56-02/baseline_moe_dag', format='svg', cleanup=True)
    
    return '/home/wzc/data/file-share/2025-09-09-10-56-02/baseline_moe_dag.dot'

if __name__ == '__main__':
    create_baseline_dag()