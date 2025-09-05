#!/usr/bin/env python3

import graphviz

def generate_proposed_dag():
    """
    Generate proposed DAG for 64 GPUs with 1 expert per GPU (EP=64)
    """
    dot = graphviz.Digraph('proposed_moe_dag', comment='Proposed MoE Deployment')
    dot.attr(rankdir='TB', size='30,40')
    
    # Define colors for different GPU groups
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink', 
              'lightgray', 'lightcyan', 'lightsteelblue', 'gold', 'orange',
              'plum', 'salmon', 'tan', 'thistle', 'wheat', 'yellowgreen']
    
    # Model parameters
    batch_size = 1024
    seq_len = 10000
    hidden_dim = 8192  # 16 heads * 512 dim
    ffn_hidden = 32768
    num_layers = 4
    num_experts = 64  # 64 experts per layer (EP=64)
    
    # GPU configuration
    total_gpus = 64
    experts_per_gpu = 1  # One expert per GPU
    
    # Create input node
    dot.node('input', f'Total Input\\nBatch: {batch_size}×{seq_len}\\nHidden: {hidden_dim}', 
             shape='ellipse', style='filled', fillcolor='white')
    
    # Process through all layers
    for layer in range(num_layers):
        # Create layer input
        layer_input = f'layer_{layer}_input'
        if layer == 0:
            dot.edge('input', layer_input)
        else:
            prev_layer = f'layer_{layer-1}_output'
            dot.edge(prev_layer, layer_input)
        
        # Layer norm (replicated across all GPUs for routing)
        ln_nodes = []
        for gpu_id in range(total_gpus):
            ln_node = f'layer_{layer}_ln_gpu_{gpu_id}'
            dot.node(ln_node, f'Layer Norm\\nIn: ({batch_size}, {seq_len}, {hidden_dim})\\nOut: ({batch_size}, {seq_len}, {hidden_dim})\\nGPU: {gpu_id}', 
                     shape='rectangle', style='filled', fillcolor='lightgray')
            if layer == 0:
                dot.edge('input', ln_node)
            else:
                prev_layer_out = f'layer_{layer-1}_output'
                dot.edge(prev_layer_out, ln_node)
            ln_nodes.append(ln_node)
        
        # Multi-head attention with expert placement
        # Since we have 64 GPUs, we need to distribute attention computation
        # Let's use 8 GPUs for attention, rest for experts
        attn_gpus = list(range(8))  # First 8 GPUs for attention
        
        # QKV projection (distributed)
        qkv_nodes = []
        heads_per_gpu = 16 // 8  # 2 heads per GPU
        for gpu_idx, gpu_id in enumerate(attn_gpus):
            qkv_node = f'layer_{layer}_qkv_gpu_{gpu_id}'
            dot.node(qkv_node, f'QKV Projection\\nIn: ({batch_size}, {seq_len}, {hidden_dim})\\nOut: ({batch_size}, {seq_len}, {hidden_dim//8 * 3})\\nGPU: {gpu_id}', 
                     shape='rectangle', style='filled', fillcolor=colors[gpu_id])
            dot.edge(ln_nodes[gpu_id], qkv_node)
            qkv_nodes.append(qkv_node)
        
        # Attention computation
        attn_nodes = []
        for gpu_idx, gpu_id in enumerate(attn_gpus):
            attn_node = f'layer_{layer}_attn_gpu_{gpu_id}'
            dot.node(attn_node, f'MHA\\nHeads: {heads_per_gpu}\\nIn: ({batch_size}, {seq_len}, {hidden_dim//8 * 3})\\nOut: ({batch_size}, {seq_len}, {hidden_dim//8})\\nGPU: {gpu_id}', 
                     shape='rectangle', style='filled', fillcolor=colors[gpu_id])
            dot.edge(qkv_nodes[gpu_idx], attn_node)
            attn_nodes.append(attn_node)
        
        # Output projection
        attn_out_nodes = []
        for gpu_idx, gpu_id in enumerate(attn_gpus):
            out_node = f'layer_{layer}_attn_out_gpu_{gpu_id}'
            dot.node(out_node, f'Attn Output\\nIn: ({batch_size}, {seq_len}, {hidden_dim//8})\\nOut: ({batch_size}, {seq_len}, {hidden_dim//8})\\nGPU: {gpu_id}', 
                     shape='rectangle', style='filled', fillcolor=colors[gpu_id])
            dot.edge(attn_nodes[gpu_idx], out_node)
            attn_out_nodes.append(out_node)
        
        # All-reduce for attention output
        attn_reduce = f'layer_{layer}_attn_reduce'
        dot.node(attn_reduce, f'All-Reduce Sum\\nIn: 8×({batch_size}, {seq_len}, {hidden_dim//8})\\nOut: ({batch_size}, {seq_len}, {hidden_dim})\\nGPUs: 0-7', 
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
        for out_node in attn_out_nodes:
            dot.edge(out_node, attn_reduce)
        
        # Broadcast attention output to all GPUs
        attn_broadcast = f'layer_{layer}_attn_broadcast'
        dot.node(attn_broadcast, f'Broadcast\\nIn: ({batch_size}, {seq_len}, {hidden_dim})\\nOut: 64×({batch_size}, {seq_len}, {hidden_dim})\\nTo: All GPUs', 
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
        dot.edge(attn_reduce, attn_broadcast)
        
        # Residual connections on all GPUs
        attn_residual_nodes = []
        for gpu_id in range(total_gpus):
            residual_node = f'layer_{layer}_attn_residual_gpu_{gpu_id}'
            dot.node(residual_node, f'Residual Add\\nIn: ({batch_size}, {seq_len}, {hidden_dim}) + ({batch_size}, {seq_len}, {hidden_dim})\\nOut: ({batch_size}, {seq_len}, {hidden_dim})\\nGPU: {gpu_id}', 
                     shape='parallelogram', style='filled', fillcolor='lightgreen')
            dot.edge(attn_broadcast, residual_node)
            dot.edge(ln_nodes[gpu_id], residual_node)
            attn_residual_nodes.append(residual_node)
        
        # Post-attention layer norm on all GPUs
        post_ln_nodes = []
        for gpu_id in range(total_gpus):
            post_ln = f'layer_{layer}_post_ln_gpu_{gpu_id}'
            dot.node(post_ln, f'Post-Attention LN\\nIn: ({batch_size}, {seq_len}, {hidden_dim})\\nOut: ({batch_size}, {seq_len}, {hidden_dim})\\nGPU: {gpu_id}', 
                     shape='rectangle', style='filled', fillcolor='lightgray')
            dot.edge(attn_residual_nodes[gpu_id], post_ln)
            post_ln_nodes.append(post_ln)
        
        # Gate computation on all GPUs (for routing)
        gate_nodes = []
        for gpu_id in range(total_gpus):
            gate_node = f'layer_{layer}_gate_gpu_{gpu_id}'
            dot.node(gate_node, f'Gate\\nIn: ({batch_size}, {seq_len}, {hidden_dim})\\nOut: ({batch_size}, {seq_len}, {num_experts})\\nGPU: {gpu_id}', 
                     shape='parallelogram', style='filled', fillcolor=colors[gpu_id % len(colors)])
            dot.edge(post_ln_nodes[gpu_id], gate_node)
            gate_nodes.append(gate_node)
        
        # Expert computation - 1 expert per GPU
        expert_outputs = []
        for gpu_id in range(total_gpus):
            expert_node = f'layer_{layer}_expert_{gpu_id}_gpu_{gpu_id}'
            dot.node(expert_node, f'Expert {gpu_id}\\nIn: ({batch_size}, {seq_len}, {hidden_dim})\\nOut: ({batch_size}, {seq_len}, {hidden_dim})\\nGPU: {gpu_id}', 
                     shape='rectangle', style='filled', fillcolor=colors[gpu_id % len(colors)])
            
            # Dashed line from gate to expert (routing decision)
            # Each gate computes routing for all experts
            for gate_gpu in range(total_gpus):
                dot.edge(gate_nodes[gate_gpu], expert_node, style='dashed', label=f'route_from_{gate_gpu}')
            
            # Data flow from post layer norm
            dot.edge(post_ln_nodes[gpu_id], expert_node)
            expert_outputs.append(expert_node)
        
        # Expert aggregation - collect outputs from all experts
        expert_agg = f'layer_{layer}_expert_agg'
        dot.node(expert_agg, f'Expert Aggregation\\nIn: 64×({batch_size}, {seq_len}, {hidden_dim})\\nOut: ({batch_size}, {seq_len}, {hidden_dim})\\nAll-to-All Reduce', 
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
        for expert_out in expert_outputs:
            dot.edge(expert_out, expert_agg)
        
        # Broadcast aggregated result back to all GPUs
        expert_broadcast = f'layer_{layer}_expert_broadcast'
        dot.node(expert_broadcast, f'Broadcast\\nIn: ({batch_size}, {seq_len}, {hidden_dim})\\nOut: 64×({batch_size}, {seq_len}, {hidden_dim})\\nTo: All GPUs', 
                 shape='parallelogram', style='filled', fillcolor='lightyellow')
        dot.edge(expert_agg, expert_broadcast)
        
        # Final residual connections on all GPUs
        final_residual_nodes = []
        for gpu_id in range(total_gpus):
            final_residual = f'layer_{layer}_final_residual_gpu_{gpu_id}'
            dot.node(final_residual, f'Final Residual\\nIn: ({batch_size}, {seq_len}, {hidden_dim}) + ({batch_size}, {seq_len}, {hidden_dim})\\nOut: ({batch_size}, {seq_len}, {hidden_dim})\\nGPU: {gpu_id}', 
                     shape='parallelogram', style='filled', fillcolor='lightgreen')
            dot.edge(expert_broadcast, final_residual)
            dot.edge(attn_residual_nodes[gpu_id], final_residual)
            final_residual_nodes.append(final_residual)
        
        # Reduce to single output (or keep distributed)
        layer_output = f'layer_{layer}_output'
        dot.node(layer_output, f'Layer {layer} Output\\n({batch_size}, {seq_len}, {hidden_dim})\\nDistributed across GPUs', 
                 shape='ellipse', style='filled', fillcolor='white')
        
        # Connect all final residuals to layer output
        for gpu_id in range(total_gpus):
            dot.edge(final_residual_nodes[gpu_id], layer_output)
    
    # Final output
    dot.node('output', f'Total Output\\nBatch: {batch_size}×{seq_len}\\nHidden: {hidden_dim}', 
             shape='ellipse', style='filled', fillcolor='white')
    
    # Connect last layer to output
    last_layer = f'layer_{num_layers-1}_output'
    dot.edge(last_layer, 'output')
    
    return dot

if __name__ == '__main__':
    dag = generate_proposed_dag()
    dag.render('/home/wzc/data/file-share/2025-09-05-11-51-32/proposed_moe_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-05-11-51-32/proposed_moe_dag.dot')
    print("Proposed DAG generated successfully")