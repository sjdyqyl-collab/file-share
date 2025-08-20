#!/usr/bin/env python3

import graphviz

def create_proposed_dag():
    """
    Create DAG for proposed deployment: 64 GPUs, 1 expert per GPU, EP=16 per layer
    """
    dot = graphviz.Digraph(comment='Proposed Cross-Node Expert Parallelism')
    dot.attr(rankdir='TB', size='30,40')
    
    # Global attributes
    dot.attr('node', fontname='Arial', fontsize='9')
    dot.attr('edge', fontname='Arial', fontsize='7')
    
    # Input layer
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input (Batch: 1024 tokens)', style='rounded')
        c.node('input', 'Input Tokens\n[1024, 8192]', shape='ellipse')
    
    # Process each layer (0-3)
    for layer_id in range(4):
        with dot.subgraph(name=f'cluster_layer{layer_id}') as layer:
            layer.attr(label=f'Layer {layer_id} - EP=16 (64 GPUs)', style='rounded', color='blue')
            
            # MHA across all GPUs (can use tensor parallelism if needed)
            with layer.subgraph(name=f'cluster_layer{layer_id}_mha') as mha:
                mha.attr(label='Multi-Head Attention\nTP across selected GPUs', style='dotted')
                
                # MHA components
                mha.node(f'l{layer_id}_mha_qkv', f'MHA QKV Linear\n[1024, 8192] → [1024, 3×16×512]\nTP across GPUs', shape='rectangle')
                mha.node(f'l{layer_id}_mha_attn', f'MHA Attention\n[1024, 16×512]\nTP across GPUs', shape='rectangle')
                mha.node(f'l{layer_id}_mha_out', f'MHA Output Linear\n[1024, 8192] → [1024, 8192]\nTP across GPUs', shape='rectangle')
                mha.node(f'l{layer_id}_mha_res', f'Residual Add\n[1024, 8192] + [1024, 8192]', shape='parallelogram')
            
            # Expert distribution - 16 experts across 64 GPUs (4 GPUs per expert group)
            # Since we have 16 experts and 64 GPUs, we can place 1 expert per GPU
            # but use 4 GPUs per expert for tensor parallelism if needed
            
            # Global gate
            layer.node(f'l{layer_id}_global_gate', f'Global Gate\n[1024, 8192] → [1024, 16]\nAll GPUs', shape='parallelogram')
            
            # Token routing and expert computation
            for expert_id in range(16):
                gpu_start = layer_id * 16 + expert_id * 4  # 4 GPUs per expert
                
                with layer.subgraph(name=f'cluster_expert{layer_id}_{expert_id}') as expert:
                    expert.attr(label=f'Expert {expert_id}\nGPUs {gpu_start}-{gpu_start+3}', style='rounded')
                    
                    # Token routing to this expert
                    expert.node(f'l{layer_id}_route_{expert_id}', f'Token Routing\nSelect tokens for Expert {expert_id}\nGPU {gpu_start}', shape='parallelogram')
                    
                    # Token gathering
                    expert.node(f'l{layer_id}_gather_{expert_id}', f'Gather Tokens\nDynamic batch size\nAll GPUs → GPU {gpu_start}', shape='ellipse')
                    
                    # Expert MLP computation (can use tensor parallelism)
                    expert.node(f'l{layer_id}_exp{expert_id}_linear1', f'Expert {expert_id} Linear 1\n[dynamic, 8192] → [dynamic, 32768]\nTP across GPUs {gpu_start}-{gpu_start+3}', shape='rectangle')
                    expert.node(f'l{layer_id}_exp{expert_id}_gelu', f'GELU Activation\n[dynamic, 32768]\nGPUs {gpu_start}-{gpu_start+3}', shape='rectangle')
                    expert.node(f'l{layer_id}_exp{expert_id}_linear2', f'Expert {expert_id} Linear 2\n[dynamic, 32768] → [dynamic, 8192]\nTP across GPUs {gpu_start}-{gpu_start+3}', shape='rectangle')
                    
                    # Token scattering back
                    expert.node(f'l{layer_id}_scatter_{expert_id}', f'Scatter Results\n[dynamic, 8192]\nGPU {gpu_start} → All GPUs', shape='ellipse')
            
            # Expert aggregation
            layer.node(f'l{layer_id}_expert_agg', f'Expert Aggregation\nCombine all expert outputs\n[1024, 8192]\nAll GPUs', shape='parallelogram')
            layer.node(f'l{layer_id}_moe_res', f'Residual Add\n[1024, 8192] + [1024, 8192]', shape='parallelogram')
    
    # Output layer
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output', style='rounded')
        c.node('output', 'Output Tokens\n[1024, 8192]', shape='ellipse')
    
    # Connections
    # Input to Layer 0
    dot.edge('input', 'l0_mha_qkv')
    dot.edge('l0_mha_qkv', 'l0_mha_attn')
    dot.edge('l0_mha_attn', 'l0_mha_out')
    dot.edge('l0_mha_out', 'l0_mha_res')
    dot.edge('input', 'l0_mha_res')  # Residual
    
    # Layer 0 MHA to MoE
    dot.edge('l0_mha_res', 'l0_global_gate')
    
    # Global gate to each expert
    for expert_id in range(16):
        dot.edge('l0_global_gate', f'l0_route_{expert_id}', style='dashed', label=f'select expert {expert_id}')
        dot.edge('l0_mha_res', f'l0_gather_{expert_id}', label='tokens')
        dot.edge(f'l0_gather_{expert_id}', f'l0_exp{expert_id}_linear1')
        dot.edge(f'l0_exp{expert_id}_linear1', f'l0_exp{expert_id}_gelu')
        dot.edge(f'l0_exp{expert_id}_gelu', f'l0_exp{expert_id}_linear2')
        dot.edge(f'l0_exp{expert_id}_linear2', f'l0_scatter_{expert_id}')
        dot.edge(f'l0_scatter_{expert_id}', f'l0_expert_agg')
    
    dot.edge('l0_expert_agg', 'l0_moe_res')
    dot.edge('l0_mha_res', 'l0_moe_res')  # Residual
    
    # Layer 0 to Layer 1
    dot.edge('l0_moe_res', 'l1_mha_qkv')
    
    # Layer 1 connections (similar to layer 0)
    dot.edge('l1_mha_qkv', 'l1_mha_attn')
    dot.edge('l1_mha_attn', 'l1_mha_out')
    dot.edge('l1_mha_out', 'l1_mha_res')
    dot.edge('l0_moe_res', 'l1_mha_res')  # Residual
    
    dot.edge('l1_mha_res', 'l1_global_gate')
    for expert_id in range(16):
        dot.edge('l1_global_gate', f'l1_route_{expert_id}', style='dashed')
        dot.edge('l1_mha_res', f'l1_gather_{expert_id}')
        dot.edge(f'l1_gather_{expert_id}', f'l1_exp{expert_id}_linear1')
        dot.edge(f'l1_exp{expert_id}_linear1', f'l1_exp{expert_id}_gelu')
        dot.edge(f'l1_exp{expert_id}_gelu', f'l1_exp{expert_id}_linear2')
        dot.edge(f'l1_exp{expert_id}_linear2', f'l1_scatter_{expert_id}')
        dot.edge(f'l1_scatter_{expert_id}', f'l1_expert_agg')
    
    dot.edge('l1_expert_agg', 'l1_moe_res')
    dot.edge('l1_mha_res', 'l1_moe_res')  # Residual
    
    # Layer 1 to Layer 2
    dot.edge('l1_moe_res', 'l2_mha_qkv')
    
    # Layer 2 connections
    dot.edge('l2_mha_qkv', 'l2_mha_attn')
    dot.edge('l2_mha_attn', 'l2_mha_out')
    dot.edge('l2_mha_out', 'l2_mha_res')
    dot.edge('l1_moe_res', 'l2_mha_res')  # Residual
    
    dot.edge('l2_mha_res', 'l2_global_gate')
    for expert_id in range(16):
        dot.edge('l2_global_gate', f'l2_route_{expert_id}', style='dashed')
        dot.edge('l2_mha_res', f'l2_gather_{expert_id}')
        dot.edge(f'l2_gather_{expert_id}', f'l2_exp{expert_id}_linear1')
        dot.edge(f'l2_exp{expert_id}_linear1', f'l2_exp{expert_id}_gelu')
        dot.edge(f'l2_exp{expert_id}_gelu', f'l2_exp{expert_id}_linear2')
        dot.edge(f'l2_exp{expert_id}_linear2', f'l2_scatter_{expert_id}')
        dot.edge(f'l2_scatter_{expert_id}', f'l2_expert_agg')
    
    dot.edge('l2_expert_agg', 'l2_moe_res')
    dot.edge('l2_mha_res', 'l2_moe_res')  # Residual
    
    # Layer 2 to Layer 3
    dot.edge('l2_moe_res', 'l3_mha_qkv')
    
    # Layer 3 connections
    dot.edge('l3_mha_qkv', 'l3_mha_attn')
    dot.edge('l3_mha_attn', 'l3_mha_out')
    dot.edge('l3_mha_out', 'l3_mha_res')
    dot.edge('l2_moe_res', 'l3_mha_res')  # Residual
    
    dot.edge('l3_mha_res', 'l3_global_gate')
    for expert_id in range(16):
        dot.edge('l3_global_gate', f'l3_route_{expert_id}', style='dashed')
        dot.edge('l3_mha_res', f'l3_gather_{expert_id}')
        dot.edge(f'l3_gather_{expert_id}', f'l3_exp{expert_id}_linear1')
        dot.edge(f'l3_exp{expert_id}_linear1', f'l3_exp{expert_id}_gelu')
        dot.edge(f'l3_exp{expert_id}_gelu', f'l3_exp{expert_id}_linear2')
        dot.edge(f'l3_exp{expert_id}_linear2', f'l3_scatter_{expert_id}')
        dot.edge(f'l3_scatter_{expert_id}', f'l3_expert_agg')
    
    dot.edge('l3_expert_agg', 'l3_moe_res')
    dot.edge('l3_mha_res', 'l3_moe_res')  # Residual
    
    # Final output
    dot.edge('l3_moe_res', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_proposed_dag()
    dag.render('/home/wzc/data/file-share/submission/proposed_moe_deployment', format='svg', cleanup=True)
    print("Proposed DAG generated successfully")