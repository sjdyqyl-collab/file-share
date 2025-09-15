#!/usr/bin/env python3

import graphviz

def create_proposed_dag():
    """Generate proposed DAG with EP=64, 1 expert per GPU, PP=4"""
    dot = graphviz.Digraph('MoE_Proposed_EP64_PP4', format='svg')
    dot.attr(rankdir='TB', ranksep='1.5', nodesep='0.8')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='yellow')  # Routing/Aggregation
    
    # Global dimensions
    batch_size = 1024
    seq_len = 10000
    hidden_dim = 8192
    ffn_hidden = 32768
    num_heads = 16
    head_dim = 512
    
    # Input node
    dot.node('input', f'Total Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: all', 
             shape='ellipse', fillcolor='lightblue')
    
    # Split for pipeline parallelism across 4 stages
    dot.node('split_pp', f'Split for 4-Stage Pipeline\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: all', 
             shape='parallelogram', fillcolor='yellow')
    
    # Process each layer separately (4 layers = 4 pipeline stages)
    for layer in range(4):
        stage_gpus = list(range(layer * 16, (layer + 1) * 16))
        
        # Layer input
        dot.node(f'layer{layer}_input', f'Layer {layer} Input\\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {stage_gpus[0]}-{stage_gpus[-1]}', 
                 shape='ellipse', fillcolor='lightblue')
        
        # Multi-head attention (replicated across stage)
        dot.node(f'layer{layer}_mha_qkv', f'QKV Linear\\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: {stage_gpus[0]}-{stage_gpus[-1]} (replicated)', 
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f'layer{layer}_mha_attn', f'Multi-Head Attention\\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {stage_gpus[0]}-{stage_gpus[-1]} (replicated)', 
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f'layer{layer}_mha_proj', f'Output Projection\\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {stage_gpus[0]}-{stage_gpus[-1]} (replicated)', 
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f'layer{layer}_mha_residual', f'Residual Add\\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {stage_gpus[0]}-{stage_gpus[-1]}', 
                 shape='parallelogram', fillcolor='yellow')
        
        # Distributed gate for expert routing
        dot.node(f'layer{layer}_gate', f'Top-2 Gating\\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, expert_indices]\\nGPU: {stage_gpus[0]}-{stage_gpus[-1]} (distributed)', 
                 shape='parallelogram', fillcolor='yellow')
        
        # Token routing - split tokens by expert destination
        dot.node(f'layer{layer}_token_split', f'Token Split by Expert\\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [variable_batch, variable_seq, hidden={hidden_dim}]\\nGPU: {stage_gpus[0]}-{stage_gpus[-1]}', 
                 shape='parallelogram', fillcolor='yellow')
        
        # Async communication for token routing
        dot.node(f'layer{layer}_async_send', f'Async Token Send\\nInput: [variable_batch, variable_seq, hidden={hidden_dim}]\\nOutput: [variable_batch, variable_seq, hidden={hidden_dim}]\\nGPU: cross-device routing', 
                 shape='parallelogram', fillcolor='yellow')
        
        # One expert per GPU (16 experts per layer)
        for gpu_idx, gpu in enumerate(stage_gpus):
            expert_id = layer * 16 + gpu_idx
            dot.node(f'layer{layer}_expert{expert_id}', f'Expert {expert_id}\\nInput: [variable_batch, variable_seq, hidden={hidden_dim}]\\nOutput: [variable_batch, variable_seq, hidden={hidden_dim}]\\nGPU: {gpu}', 
                     shape='rectangle', fillcolor='lightgreen')
        
        # Async communication for token return
        dot.node(f'layer{layer}_async_recv', f'Async Token Receive\\nInput: [variable_batch, variable_seq, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: cross-device routing', 
                 shape='parallelogram', fillcolor='yellow')
        
        # Expert aggregation
        dot.node(f'layer{layer}_expert_agg', f'Expert Aggregation\\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {stage_gpus[0]}-{stage_gpus[-1]}', 
                 shape='parallelogram', fillcolor='yellow')
        
        dot.node(f'layer{layer}_ffn_residual', f'Residual Add\\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {stage_gpus[0]}-{stage_gpus[-1]}', 
                 shape='parallelogram', fillcolor='yellow')
        
        # Layer output
        dot.node(f'layer{layer}_output', f'Layer {layer} Output\\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {stage_gpus[0]}-{stage_gpus[-1]}', 
                 shape='ellipse', fillcolor='lightblue')
        
        # Pipeline communication to next stage
        if layer < 3:
            next_stage_gpus = list(range((layer + 1) * 16, (layer + 2) * 16))
            dot.node(f'pipeline_send{layer}', f'Send to Stage {layer+1}\\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {stage_gpus[-1]} → {next_stage_gpus[0]}', 
                     shape='parallelogram', fillcolor='yellow')
    
    # Final aggregation
    dot.node('aggregate_pp', f'Aggregate Pipeline\\nInput: [batch_size={batch_size//4}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: all', 
             shape='parallelogram', fillcolor='yellow')
    
    # Output
    dot.node('output', f'Total Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: all', 
             shape='ellipse', fillcolor='lightblue')
    
    # Create edges
    dot.edge('input', 'split_pp')
    dot.edge('split_pp', 'layer0_input')
    
    # Layer 0 connections
    dot.edge('layer0_input', 'layer0_mha_qkv')
    dot.edge('layer0_mha_qkv', 'layer0_mha_attn')
    dot.edge('layer0_mha_attn', 'layer0_mha_proj')
    dot.edge('layer0_input', 'layer0_mha_residual')  # residual connection
    dot.edge('layer0_mha_proj', 'layer0_mha_residual')
    dot.edge('layer0_mha_residual', 'layer0_gate')
    dot.edge('layer0_gate', 'layer0_token_split')
    dot.edge('layer0_token_split', 'layer0_async_send')
    
    # Expert connections for layer 0
    for gpu_idx, gpu in enumerate(range(16)):
        expert_id = gpu_idx
        dot.edge('layer0_async_send', f'layer0_expert{expert_id}', style='dashed')
        dot.edge(f'layer0_expert{expert_id}', 'layer0_async_recv')
    
    dot.edge('layer0_async_recv', 'layer0_expert_agg')
    dot.edge('layer0_expert_agg', 'layer0_ffn_residual')
    dot.edge('layer0_mha_residual', 'layer0_ffn_residual')  # residual connection
    dot.edge('layer0_ffn_residual', 'layer0_output')
    dot.edge('layer0_output', 'pipeline_send0')
    dot.edge('pipeline_send0', 'layer1_input')
    
    # Layer 1 connections
    dot.edge('layer1_input', 'layer1_mha_qkv')
    dot.edge('layer1_mha_qkv', 'layer1_mha_attn')
    dot.edge('layer1_mha_attn', 'layer1_mha_proj')
    dot.edge('layer1_input', 'layer1_mha_residual')
    dot.edge('layer1_mha_proj', 'layer1_mha_residual')
    dot.edge('layer1_mha_residual', 'layer1_gate')
    dot.edge('layer1_gate', 'layer1_token_split')
    dot.edge('layer1_token_split', 'layer1_async_send')
    
    for gpu_idx, gpu in enumerate(range(16, 32)):
        expert_id = 16 + gpu_idx
        dot.edge('layer1_async_send', f'layer1_expert{expert_id}', style='dashed')
        dot.edge(f'layer1_expert{expert_id}', 'layer1_async_recv')
    
    dot.edge('layer1_async_recv', 'layer1_expert_agg')
    dot.edge('layer1_expert_agg', 'layer1_ffn_residual')
    dot.edge('layer1_mha_residual', 'layer1_ffn_residual')
    dot.edge('layer1_ffn_residual', 'layer1_output')
    dot.edge('layer1_output', 'pipeline_send1')
    dot.edge('pipeline_send1', 'layer2_input')
    
    # Layer 2 connections
    dot.edge('layer2_input', 'layer2_mha_qkv')
    dot.edge('layer2_mha_qkv', 'layer2_mha_attn')
    dot.edge('layer2_mha_attn', 'layer2_mha_proj')
    dot.edge('layer2_input', 'layer2_mha_residual')
    dot.edge('layer2_mha_proj', 'layer2_mha_residual')
    dot.edge('layer2_mha_residual', 'layer2_gate')
    dot.edge('layer2_gate', 'layer2_token_split')
    dot.edge('layer2_token_split', 'layer2_async_send')
    
    for gpu_idx, gpu in enumerate(range(32, 48)):
        expert_id = 32 + gpu_idx
        dot.edge('layer2_async_send', f'layer2_expert{expert_id}', style='dashed')
        dot.edge(f'layer2_expert{expert_id}', 'layer2_async_recv')
    
    dot.edge('layer2_async_recv', 'layer2_expert_agg')
    dot.edge('layer2_expert_agg', 'layer2_ffn_residual')
    dot.edge('layer2_mha_residual', 'layer2_ffn_residual')
    dot.edge('layer2_ffn_residual', 'layer2_output')
    dot.edge('layer2_output', 'pipeline_send2')
    dot.edge('pipeline_send2', 'layer3_input')
    
    # Layer 3 connections
    dot.edge('layer3_input', 'layer3_mha_qkv')
    dot.edge('layer3_mha_qkv', 'layer3_mha_attn')
    dot.edge('layer3_mha_attn', 'layer3_mha_proj')
    dot.edge('layer3_input', 'layer3_mha_residual')
    dot.edge('layer3_mha_proj', 'layer3_mha_residual')
    dot.edge('layer3_mha_residual', 'layer3_gate')
    dot.edge('layer3_gate', 'layer3_token_split')
    dot.edge('layer3_token_split', 'layer3_async_send')
    
    for gpu_idx, gpu in enumerate(range(48, 64)):
        expert_id = 48 + gpu_idx
        dot.edge('layer3_async_send', f'layer3_expert{expert_id}', style='dashed')
        dot.edge(f'layer3_expert{expert_id}', 'layer3_async_recv')
    
    dot.edge('layer3_async_recv', 'layer3_expert_agg')
    dot.edge('layer3_expert_agg', 'layer3_ffn_residual')
    dot.edge('layer3_mha_residual', 'layer3_ffn_residual')
    dot.edge('layer3_ffn_residual', 'layer3_output')
    dot.edge('layer3_output', 'aggregate_pp')
    dot.edge('aggregate_pp', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_proposed_dag()
    dag.render('/home/wzc/data/file-share/2025-09-12-16-07-56/proposed_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-12-16-07-56/proposed_dag.dot')