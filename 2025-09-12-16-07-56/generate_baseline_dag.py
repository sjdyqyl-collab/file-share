#!/usr/bin/env python3

import graphviz

def create_baseline_dag():
    """Generate baseline DAG with TP=8, PP=2, 4 experts per GPU"""
    dot = graphviz.Digraph('MoE_Baseline_TP8_PP2', format='svg')
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
    
    # Split for pipeline parallelism
    dot.node('split_pp', f'Split for Pipeline\\nInput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: all', 
             shape='parallelogram', fillcolor='yellow')
    
    # Stage 0 (GPUs 0-7)
    for layer in [0, 1]:
        # Layer input
        dot.node(f'layer{layer}_input', f'Layer {layer} Input\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: 0-7', 
                 shape='ellipse', fillcolor='lightblue')
        
        # Multi-head attention
        dot.node(f'layer{layer}_mha_qkv', f'QKV Linear\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim//8}]\\nGPU: 0-7 (TP=8)', 
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f'layer{layer}_mha_attn', f'Multi-Head Attention\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim//8}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim//8}]\\nGPU: 0-7 (TP=8)', 
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f'layer{layer}_mha_concat', f'Concat Heads\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim//8}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: 0-7 (TP=8)', 
                 shape='parallelogram', fillcolor='yellow')
        
        dot.node(f'layer{layer}_mha_proj', f'Output Projection\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: 0-7 (TP=8)', 
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f'layer{layer}_mha_residual', f'Residual Add\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: 0-7', 
                 shape='parallelogram', fillcolor='yellow')
        
        # Gate
        dot.node(f'layer{layer}_gate', f'Top-2 Gating\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, expert_indices]\\nGPU: 0-7 (replicated)', 
                 shape='parallelogram', fillcolor='yellow')
        
        # Experts (4 per GPU)
        for gpu in range(8):
            for expert in range(4):
                expert_id = layer * 32 + gpu * 4 + expert
                dot.node(f'layer{layer}_expert{expert_id}', f'Expert {expert_id}\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {gpu}', 
                         shape='rectangle', fillcolor='lightgreen')
        
        # Expert aggregation
        dot.node(f'layer{layer}_expert_agg', f'Expert Aggregation\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: 0-7', 
                 shape='parallelogram', fillcolor='yellow')
        
        dot.node(f'layer{layer}_ffn_residual', f'Residual Add\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: 0-7', 
                 shape='parallelogram', fillcolor='yellow')
        
        # Layer output
        dot.node(f'layer{layer}_output', f'Layer {layer} Output\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: 0-7', 
                 shape='ellipse', fillcolor='lightblue')
    
    # Pipeline communication
    dot.node('pipeline_send', f'Send to Stage 1\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: 7 → 8', 
             shape='parallelogram', fillcolor='yellow')
    
    # Stage 1 (GPUs 8-15)
    for layer in [2, 3]:
        actual_layer = layer
        # Similar structure as Stage 0
        dot.node(f'layer{actual_layer}_input', f'Layer {actual_layer} Input\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: 8-15', 
                 shape='ellipse', fillcolor='lightblue')
        
        # Multi-head attention
        dot.node(f'layer{actual_layer}_mha_qkv', f'QKV Linear\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim//8}]\\nGPU: 8-15 (TP=8)', 
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f'layer{actual_layer}_mha_attn', f'Multi-Head Attention\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim//8}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim//8}]\\nGPU: 8-15 (TP=8)', 
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f'layer{actual_layer}_mha_concat', f'Concat Heads\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim//8}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: 8-15 (TP=8)', 
                 shape='parallelogram', fillcolor='yellow')
        
        dot.node(f'layer{actual_layer}_mha_proj', f'Output Projection\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: 8-15 (TP=8)', 
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f'layer{actual_layer}_mha_residual', f'Residual Add\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: 8-15', 
                 shape='parallelogram', fillcolor='yellow')
        
        # Gate
        dot.node(f'layer{actual_layer}_gate', f'Top-2 Gating\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, expert_indices]\\nGPU: 8-15 (replicated)', 
                 shape='parallelogram', fillcolor='yellow')
        
        # Experts (4 per GPU)
        for gpu in range(8, 16):
            for expert in range(4):
                expert_id = actual_layer * 32 + (gpu-8) * 4 + expert
                dot.node(f'layer{actual_layer}_expert{expert_id}', f'Expert {expert_id}\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: {gpu}', 
                         shape='rectangle', fillcolor='lightgreen')
        
        # Expert aggregation
        dot.node(f'layer{actual_layer}_expert_agg', f'Expert Aggregation\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: 8-15', 
                 shape='parallelogram', fillcolor='yellow')
        
        dot.node(f'layer{actual_layer}_ffn_residual', f'Residual Add\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: 8-15', 
                 shape='parallelogram', fillcolor='yellow')
        
        # Layer output
        dot.node(f'layer{actual_layer}_output', f'Layer {actual_layer} Output\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: 8-15', 
                 shape='ellipse', fillcolor='lightblue')
    
    # Final aggregation
    dot.node('aggregate_pp', f'Aggregate Pipeline\\nInput: [batch_size={batch_size//2}, seq_len={seq_len}, hidden={hidden_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, hidden={hidden_dim}]\\nGPU: all', 
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
    dot.edge('layer0_mha_attn', 'layer0_mha_concat')
    dot.edge('layer0_mha_concat', 'layer0_mha_proj')
    dot.edge('layer0_input', 'layer0_mha_residual')  # residual connection
    dot.edge('layer0_mha_proj', 'layer0_mha_residual')
    dot.edge('layer0_mha_residual', 'layer0_gate')
    
    # Expert connections for layer 0
    for gpu in range(8):
        for expert in range(4):
            expert_id = gpu * 4 + expert
            dot.edge('layer0_gate', f'layer0_expert{expert_id}', style='dashed')
    
    # Collect expert outputs
    for gpu in range(8):
        for expert in range(4):
            expert_id = gpu * 4 + expert
            dot.edge(f'layer0_expert{expert_id}', 'layer0_expert_agg')
    
    dot.edge('layer0_expert_agg', 'layer0_ffn_residual')
    dot.edge('layer0_mha_residual', 'layer0_ffn_residual')  # residual connection
    dot.edge('layer0_ffn_residual', 'layer0_output')
    dot.edge('layer0_output', 'layer1_input')
    
    # Layer 1 connections (similar to layer 0)
    dot.edge('layer1_input', 'layer1_mha_qkv')
    dot.edge('layer1_mha_qkv', 'layer1_mha_attn')
    dot.edge('layer1_mha_attn', 'layer1_mha_concat')
    dot.edge('layer1_mha_concat', 'layer1_mha_proj')
    dot.edge('layer1_input', 'layer1_mha_residual')
    dot.edge('layer1_mha_proj', 'layer1_mha_residual')
    dot.edge('layer1_mha_residual', 'layer1_gate')
    
    for gpu in range(8):
        for expert in range(4):
            expert_id = 32 + gpu * 4 + expert
            dot.edge('layer1_gate', f'layer1_expert{expert_id}', style='dashed')
    
    for gpu in range(8):
        for expert in range(4):
            expert_id = 32 + gpu * 4 + expert
            dot.edge(f'layer1_expert{expert_id}', 'layer1_expert_agg')
    
    dot.edge('layer1_expert_agg', 'layer1_ffn_residual')
    dot.edge('layer1_mha_residual', 'layer1_ffn_residual')
    dot.edge('layer1_ffn_residual', 'layer1_output')
    dot.edge('layer1_output', 'pipeline_send')
    dot.edge('pipeline_send', 'layer2_input')
    
    # Layer 2 connections
    dot.edge('layer2_input', 'layer2_mha_qkv')
    dot.edge('layer2_mha_qkv', 'layer2_mha_attn')
    dot.edge('layer2_mha_attn', 'layer2_mha_concat')
    dot.edge('layer2_mha_concat', 'layer2_mha_proj')
    dot.edge('layer2_input', 'layer2_mha_residual')
    dot.edge('layer2_mha_proj', 'layer2_mha_residual')
    dot.edge('layer2_mha_residual', 'layer2_gate')
    
    for gpu in range(8, 16):
        for expert in range(4):
            expert_id = 64 + (gpu-8) * 4 + expert
            dot.edge('layer2_gate', f'layer2_expert{expert_id}', style='dashed')
    
    for gpu in range(8, 16):
        for expert in range(4):
            expert_id = 64 + (gpu-8) * 4 + expert
            dot.edge(f'layer2_expert{expert_id}', 'layer2_expert_agg')
    
    dot.edge('layer2_expert_agg', 'layer2_ffn_residual')
    dot.edge('layer2_mha_residual', 'layer2_ffn_residual')
    dot.edge('layer2_ffn_residual', 'layer2_output')
    dot.edge('layer2_output', 'layer3_input')
    
    # Layer 3 connections
    dot.edge('layer3_input', 'layer3_mha_qkv')
    dot.edge('layer3_mha_qkv', 'layer3_mha_attn')
    dot.edge('layer3_mha_attn', 'layer3_mha_concat')
    dot.edge('layer3_mha_concat', 'layer3_mha_proj')
    dot.edge('layer3_input', 'layer3_mha_residual')
    dot.edge('layer3_mha_proj', 'layer3_mha_residual')
    dot.edge('layer3_mha_residual', 'layer3_gate')
    
    for gpu in range(8, 16):
        for expert in range(4):
            expert_id = 96 + (gpu-8) * 4 + expert
            dot.edge('layer3_gate', f'layer3_expert{expert_id}', style='dashed')
    
    for gpu in range(8, 16):
        for expert in range(4):
            expert_id = 96 + (gpu-8) * 4 + expert
            dot.edge(f'layer3_expert{expert_id}', 'layer3_expert_agg')
    
    dot.edge('layer3_expert_agg', 'layer3_ffn_residual')
    dot.edge('layer3_mha_residual', 'layer3_ffn_residual')
    dot.edge('layer3_ffn_residual', 'layer3_output')
    dot.edge('layer3_output', 'aggregate_pp')
    dot.edge('aggregate_pp', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/2025-09-12-16-07-56/baseline_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-12-16-07-56/baseline_dag.dot')