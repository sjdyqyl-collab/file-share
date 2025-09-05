#!/usr/bin/env python3

import graphviz

def generate_baseline_dag():
    """
    Generate baseline DAG for 16 GPUs with TP=8, PP=2, 4 experts per GPU
    """
    dot = graphviz.Digraph('baseline_moe_dag', comment='Baseline MoE Deployment')
    dot.attr(rankdir='TB', size='20,30')
    
    # Define colors for different GPU groups
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 
              'lightpink', 'lightgray', 'lightcyan', 'lightsteelblue']
    
    # Model parameters
    batch_size = 1024
    seq_len = 10000
    hidden_dim = 8192  # 16 heads * 512 dim
    ffn_hidden = 32768
    num_layers = 4
    num_experts = 16
    
    # GPU configuration
    total_gpus = 16
    tp_degree = 8  # Tensor parallelism across 8 GPUs
    pp_degree = 2  # Pipeline parallelism across 2 stages
    experts_per_gpu = 4
    
    # Create input node
    dot.node('input', f'Total Input\\nBatch: {batch_size}×{seq_len}\\nHidden: {hidden_dim}', 
             shape='ellipse', style='filled', fillcolor='white')
    
    # Process through pipeline stages
    for stage in range(pp_degree):
        stage_gpus = list(range(stage * 8, (stage + 1) * 8))
        
        for layer in range(num_layers // pp_degree):
            layer_id = stage * (num_layers // pp_degree) + layer
            
            # Create layer input
            layer_input = f'layer_{layer_id}_input'
            if layer_id == 0:
                dot.edge('input', layer_input)
            else:
                prev_layer = f'layer_{layer_id-1}_output'
                dot.edge(prev_layer, layer_input)
            
            # Layer norm (all GPUs)
            ln_node = f'layer_{layer_id}_ln'
            dot.node(ln_node, f'Layer Norm\\nIn: ({batch_size}, {seq_len}, {hidden_dim})\\nOut: ({batch_size}, {seq_len}, {hidden_dim})\\nGPUs: {stage_gpus}', 
                     shape='rectangle', style='filled', fillcolor='lightgray')
            dot.edge(layer_input, ln_node)
            
            # Multi-head attention with TP=8
            # Split heads across 8 GPUs
            heads_per_gpu = 16 // 8  # 2 heads per GPU
            
            # QKV projection (column parallel)
            qkv_nodes = []
            for gpu_idx, gpu_id in enumerate(stage_gpus):
                qkv_node = f'layer_{layer_id}_qkv_gpu_{gpu_id}'
                dot.node(qkv_node, f'QKV Projection\\nIn: ({batch_size}, {seq_len}, {hidden_dim})\\nOut: ({batch_size}, {seq_len}, {hidden_dim//8 * 3})\\nGPU: {gpu_id}', 
                         shape='rectangle', style='filled', fillcolor=colors[gpu_idx % len(colors)])
                dot.edge(ln_node, qkv_node)
                qkv_nodes.append(qkv_node)
            
            # Attention computation per GPU
            attn_nodes = []
            for gpu_idx, gpu_id in enumerate(stage_gpus):
                attn_node = f'layer_{layer_id}_attn_gpu_{gpu_id}'
                dot.node(attn_node, f'MHA\\nHeads: {heads_per_gpu}\\nIn: ({batch_size}, {seq_len}, {hidden_dim//8 * 3})\\nOut: ({batch_size}, {seq_len}, {hidden_dim//8})\\nGPU: {gpu_id}', 
                         shape='rectangle', style='filled', fillcolor=colors[gpu_idx % len(colors)])
                dot.edge(qkv_nodes[gpu_idx], attn_node)
                attn_nodes.append(attn_node)
            
            # Output projection (row parallel)
            attn_out_nodes = []
            for gpu_idx, gpu_id in enumerate(stage_gpus):
                out_node = f'layer_{layer_id}_attn_out_gpu_{gpu_id}'
                dot.node(out_node, f'Attn Output\\nIn: ({batch_size}, {seq_len}, {hidden_dim//8})\\nOut: ({batch_size}, {seq_len}, {hidden_dim//8})\\nGPU: {gpu_id}', 
                         shape='rectangle', style='filled', fillcolor=colors[gpu_idx % len(colors)])
                dot.edge(attn_nodes[gpu_idx], out_node)
                attn_out_nodes.append(out_node)
            
            # All-reduce for attention output
            attn_reduce = f'layer_{layer_id}_attn_reduce'
            dot.node(attn_reduce, f'All-Reduce Sum\\nIn: 8×({batch_size}, {seq_len}, {hidden_dim//8})\\nOut: ({batch_size}, {seq_len}, {hidden_dim})\\nGPUs: {stage_gpus}', 
                     shape='parallelogram', style='filled', fillcolor='lightyellow')
            for out_node in attn_out_nodes:
                dot.edge(out_node, attn_reduce)
            
            # Residual connection
            attn_residual = f'layer_{layer_id}_attn_residual'
            dot.node(attn_residual, f'Residual Add\\nIn: ({batch_size}, {seq_len}, {hidden_dim}) + ({batch_size}, {seq_len}, {hidden_dim})\\nOut: ({batch_size}, {seq_len}, {hidden_dim})\\nGPUs: {stage_gpus}', 
                     shape='parallelogram', style='filled', fillcolor='lightgreen')
            dot.edge(attn_reduce, attn_residual)
            dot.edge(ln_node, attn_residual)  # Skip connection
            
            # Post-attention layer norm
            post_ln = f'layer_{layer_id}_post_ln'
            dot.node(post_ln, f'Post-Attention LN\\nIn: ({batch_size}, {seq_len}, {hidden_dim})\\nOut: ({batch_size}, {seq_len}, {hidden_dim})\\nGPUs: {stage_gpus}', 
                     shape='rectangle', style='filled', fillcolor='lightgray')
            dot.edge(attn_residual, post_ln)
            
            # MoE layer - 4 experts per GPU, 16 total experts
            # Gate computation (replicated across GPUs)
            gate_nodes = []
            for gpu_idx, gpu_id in enumerate(stage_gpus):
                gate_node = f'layer_{layer_id}_gate_gpu_{gpu_id}'
                dot.node(gate_node, f'Gate\\nIn: ({batch_size}, {seq_len}, {hidden_dim})\\nOut: ({batch_size}, {seq_len}, {num_experts})\\nGPU: {gpu_id}', 
                         shape='parallelogram', style='filled', fillcolor=colors[gpu_idx % len(colors)])
                dot.edge(post_ln, gate_node)
                gate_nodes.append(gate_node)
            
            # Expert computation - 4 experts per GPU
            expert_outputs = []
            for gpu_idx, gpu_id in enumerate(stage_gpus):
                gpu_experts = []
                for expert_id in range(experts_per_gpu):
                    expert_global_id = gpu_idx * experts_per_gpu + expert_id
                    expert_node = f'layer_{layer_id}_expert_{expert_global_id}_gpu_{gpu_id}'
                    dot.node(expert_node, f'Expert {expert_global_id}\\nIn: ({batch_size}, {seq_len}, {hidden_dim})\\nOut: ({batch_size}, {seq_len}, {hidden_dim})\\nGPU: {gpu_id}', 
                             shape='rectangle', style='filled', fillcolor=colors[gpu_idx % len(colors)])
                    
                    # Dashed line from gate to expert (routing decision)
                    dot.edge(gate_nodes[gpu_idx], expert_node, style='dashed', label=f'route_{expert_global_id}')
                    dot.edge(post_ln, expert_node)  # Data flow
                    gpu_experts.append(expert_node)
                expert_outputs.extend(gpu_experts)
            
            # Expert aggregation
            expert_agg = f'layer_{layer_id}_expert_agg'
            dot.node(expert_agg, f'Expert Aggregation\\nIn: 16×({batch_size}, {seq_len}, {hidden_dim})\\nOut: ({batch_size}, {seq_len}, {hidden_dim})\\nGPUs: {stage_gpus}', 
                     shape='parallelogram', style='filled', fillcolor='lightyellow')
            for expert_out in expert_outputs:
                dot.edge(expert_out, expert_agg)
            
            # Final residual connection
            final_residual = f'layer_{layer_id}_final_residual'
            dot.node(final_residual, f'Final Residual\\nIn: ({batch_size}, {seq_len}, {hidden_dim}) + ({batch_size}, {seq_len}, {hidden_dim})\\nOut: ({batch_size}, {seq_len}, {hidden_dim})\\nGPUs: {stage_gpus}', 
                     shape='parallelogram', style='filled', fillcolor='lightgreen')
            dot.edge(expert_agg, final_residual)
            dot.edge(attn_residual, final_residual)  # Skip connection
            
            # Layer output
            layer_output = f'layer_{layer_id}_output'
            dot.node(layer_output, f'Layer {layer_id} Output\\n({batch_size}, {seq_len}, {hidden_dim})', 
                     shape='ellipse', style='filled', fillcolor='white')
            dot.edge(final_residual, layer_output)
    
    # Final output
    dot.node('output', f'Total Output\\nBatch: {batch_size}×{seq_len}\\nHidden: {hidden_dim}', 
             shape='ellipse', style='filled', fillcolor='white')
    
    # Connect last layer to output
    last_layer = f'layer_{num_layers-1}_output'
    dot.edge(last_layer, 'output')
    
    return dot

if __name__ == '__main__':
    dag = generate_baseline_dag()
    dag.render('/home/wzc/data/file-share/2025-09-05-11-51-32/baseline_moe_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-05-11-51-32/baseline_moe_dag.dot')
    print("Baseline DAG generated successfully")