#!/usr/bin/env python3

import graphviz

def create_proposed_dag():
    """Create proposed DAG with EP=64, 1 expert per GPU, cross-node distribution"""
    
    dot = graphviz.Digraph('Proposed_Large_EP_MoE', comment='Proposed Large EP MoE with 1 expert per GPU')
    dot.attr(rankdir='TB', size='30,40')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Communication
    dot.attr('node', shape='diamond', style='filled', fillcolor='lightcoral')  # Routing/Gating
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=1024, seq_len=10000, dim=8192]\\nGPU: All GPUs', 
             shape='ellipse', fillcolor='lightblue')
    
    # Process each layer
    for layer in range(4):
        layer_name = f"layer_{layer}"
        gpu_start = layer * 16
        
        # Multi-Head Attention (shared across layer, replicated for each expert group)
        attention_name = f"{layer_name}_attention"
        dot.node(f"{attention_name}_qkv", 
                 f"QKV Linear\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 16, 512]\\nGPU: {gpu_start}-{gpu_start+15}",
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f"{attention_name}_split", 
                 f"Split Heads\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 16, 512]\\nGPU: {gpu_start}-{gpu_start+15}",
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f"{attention_name}_matmul_qk", 
                 f"QK^T Matmul\\nInput: [1024, 10000, 16, 512]\\nOutput: [1024, 16, 10000, 10000]\\nGPU: {gpu_start}-{gpu_start+15}",
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f"{attention_name}_softmax", 
                 f"Softmax\\nInput: [1024, 16, 10000, 10000]\\nOutput: [1024, 16, 10000, 10000]\\nGPU: {gpu_start}-{gpu_start+15}",
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f"{attention_name}_matmul_v", 
                 f"Attention Output\\nInput: [1024, 16, 10000, 10000], [1024, 10000, 16, 512]\\nOutput: [1024, 10000, 16, 512]\\nGPU: {gpu_start}-{gpu_start+15}",
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f"{attention_name}_concat", 
                 f"Concat Heads\\nInput: [1024, 10000, 16, 512]\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_start}-{gpu_start+15}",
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f"{attention_name}_linear", 
                 f"Output Linear\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_start}-{gpu_start+15}",
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f"{attention_name}_residual", 
                 f"Residual Add\\nInput: [1024, 10000, 8192], [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_start}-{gpu_start+15}",
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f"{attention_name}_layernorm", 
                 f"LayerNorm\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_start}-{gpu_start+15}",
                 shape='rectangle', fillcolor='lightgreen')
        
        # Global gating for the layer
        gating_name = f"{layer_name}_gating"
        dot.node(gating_name, 
                 f"Global Gating\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 16]\\nGPU: {gpu_start}-{gpu_start+15}",
                 shape='diamond', fillcolor='lightcoral')
        
        # Token routing communication
        routing_name = f"{layer_name}_token_routing"
        dot.node(routing_name, 
                 f"Token Routing\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_start}-{gpu_start+15} → Individual GPUs",
                 shape='parallelogram', fillcolor='lightyellow')
        
        # 16 experts, one per GPU
        for expert_id in range(16):
            gpu_id = gpu_start + expert_id
            expert_name = f"{layer_name}_expert{expert_id}"
            
            # Token gathering for this expert
            gather_name = f"{expert_name}_gather"
            dot.node(gather_name, 
                     f"Gather Tokens\\nInput: [1024, 10000, 8192]\\nOutput: [batch_subset, 8192]\\nGPU: {gpu_id}",
                     shape='parallelogram', fillcolor='lightyellow')
            
            # Expert computation
            dot.node(f"{expert_name}_linear1", 
                     f"Expert {expert_id} Linear1\\nInput: [batch_subset, 8192]\\nOutput: [batch_subset, 32768]\\nGPU: {gpu_id}",
                     shape='rectangle', fillcolor='lightgreen')
            
            dot.node(f"{expert_name}_gelu", 
                     f"Expert {expert_id} GELU\\nInput: [batch_subset, 32768]\\nOutput: [batch_subset, 32768]\\nGPU: {gpu_id}",
                     shape='rectangle', fillcolor='lightgreen')
            
            dot.node(f"{expert_name}_linear2", 
                     f"Expert {expert_id} Linear2\\nInput: [batch_subset, 32768]\\nOutput: [batch_subset, 8192]\\nGPU: {gpu_id}",
                     shape='rectangle', fillcolor='lightgreen')
            
            # Token scattering back
            scatter_name = f"{expert_name}_scatter"
            dot.node(scatter_name, 
                     f"Scatter Results\\nInput: [batch_subset, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_id}",
                     shape='parallelogram', fillcolor='lightyellow')
        
        # Expert aggregation across all 16 experts
        aggregation_name = f"{layer_name}_aggregation"
        dot.node(aggregation_name, 
                 f"Expert Aggregation\\nInput: [1024, 10000, 8192] × 16\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_start}-{gpu_start+15}",
                 shape='parallelogram', fillcolor='lightyellow')
        
        # Final residual and layernorm
        dot.node(f"{layer_name}_final_residual", 
                 f"Final Residual\\nInput: [1024, 10000, 8192], [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_start}-{gpu_start+15}",
                 shape='rectangle', fillcolor='lightgreen')
        
        dot.node(f"{layer_name}_final_layernorm", 
                 f"Final LayerNorm\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_start}-{gpu_start+15}",
                 shape='rectangle', fillcolor='lightgreen')
        
        # Pipeline communication between layers
        if layer < 3:
            comm_name = f"pipeline_comm_{layer}_{layer+1}"
            next_gpu_start = (layer + 1) * 16
            dot.node(comm_name, 
                     f"Pipeline Communication\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_start+15}→{next_gpu_start}",
                     shape='parallelogram', fillcolor='lightyellow')
    
    # Output node
    dot.node('output', 'Output\\nInput: [batch_size=1024, seq_len=10000, dim=8192]\\nGPU: All GPUs', 
             shape='ellipse', fillcolor='lightblue')
    
    # Connect the DAG
    # Input to first attention
    dot.edge('input', 'layer_0_attention_qkv')
    
    # Attention connections for layer 0
    dot.edge('layer_0_attention_qkv', 'layer_0_attention_split')
    dot.edge('layer_0_attention_split', 'layer_0_attention_matmul_qk')
    dot.edge('layer_0_attention_matmul_qk', 'layer_0_attention_softmax')
    dot.edge('layer_0_attention_softmax', 'layer_0_attention_matmul_v')
    dot.edge('layer_0_attention_matmul_v', 'layer_0_attention_concat')
    dot.edge('layer_0_attention_concat', 'layer_0_attention_linear')
    dot.edge('input', 'layer_0_attention_residual')  # Residual connection
    dot.edge('layer_0_attention_linear', 'layer_0_attention_residual')
    dot.edge('layer_0_attention_residual', 'layer_0_attention_layernorm')
    
    # Gating and routing
    dot.edge('layer_0_attention_layernorm', 'layer_0_gating')
    dot.edge('layer_0_attention_layernorm', 'layer_0_token_routing')
    
    # Expert processing for layer 0
    for expert_id in range(16):
        expert_name = f'layer_0_expert{expert_id}'
        
        # Routing to expert
        dot.edge('layer_0_token_routing', f'{expert_name}_gather')
        dot.edge('layer_0_gating', f'{expert_name}_gather', style='dashed')
        
        # Expert computation
        dot.edge(f'{expert_name}_gather', f'{expert_name}_linear1')
        dot.edge(f'{expert_name}_linear1', f'{expert_name}_gelu')
        dot.edge(f'{expert_name}_gelu', f'{expert_name}_linear2')
        dot.edge(f'{expert_name}_linear2', f'{expert_name}_scatter')
        dot.edge(f'{expert_name}_scatter', 'layer_0_aggregation')
    
    # Final processing for layer 0
    dot.edge('layer_0_aggregation', 'layer_0_final_residual')
    dot.edge('layer_0_attention_layernorm', 'layer_0_final_residual')  # Residual
    dot.edge('layer_0_final_residual', 'layer_0_final_layernorm')
    dot.edge('layer_0_final_layernorm', 'pipeline_comm_0_1')
    
    # Continue with layer 1
    dot.edge('pipeline_comm_0_1', 'layer_1_attention_qkv')
    dot.edge('layer_1_attention_qkv', 'layer_1_attention_split')
    dot.edge('layer_1_attention_split', 'layer_1_attention_matmul_qk')
    dot.edge('layer_1_attention_matmul_qk', 'layer_1_attention_softmax')
    dot.edge('layer_1_attention_softmax', 'layer_1_attention_matmul_v')
    dot.edge('layer_1_attention_matmul_v', 'layer_1_attention_concat')
    dot.edge('layer_1_attention_concat', 'layer_1_attention_linear')
    dot.edge('pipeline_comm_0_1', 'layer_1_attention_residual')  # Residual
    dot.edge('layer_1_attention_linear', 'layer_1_attention_residual')
    dot.edge('layer_1_attention_residual', 'layer_1_attention_layernorm')
    
    # Gating and routing for layer 1
    dot.edge('layer_1_attention_layernorm', 'layer_1_gating')
    dot.edge('layer_1_attention_layernorm', 'layer_1_token_routing')
    
    # Expert processing for layer 1
    for expert_id in range(16):
        expert_name = f'layer_1_expert{expert_id}'
        
        dot.edge('layer_1_token_routing', f'{expert_name}_gather')
        dot.edge('layer_1_gating', f'{expert_name}_gather', style='dashed')
        dot.edge(f'{expert_name}_gather', f'{expert_name}_linear1')
        dot.edge(f'{expert_name}_linear1', f'{expert_name}_gelu')
        dot.edge(f'{expert_name}_gelu', f'{expert_name}_linear2')
        dot.edge(f'{expert_name}_linear2', f'{expert_name}_scatter')
        dot.edge(f'{expert_name}_scatter', 'layer_1_aggregation')
    
    dot.edge('layer_1_aggregation', 'layer_1_final_residual')
    dot.edge('layer_1_attention_layernorm', 'layer_1_final_residual')
    dot.edge('layer_1_final_residual', 'layer_1_final_layernorm')
    dot.edge('layer_1_final_layernorm', 'pipeline_comm_1_2')
    
    # Continue with layer 2 and 3 (similar structure)
    dot.edge('pipeline_comm_1_2', 'layer_2_attention_qkv')
    # ... (layer 2 processing)
    dot.edge('layer_2_final_layernorm', 'pipeline_comm_2_3')
    dot.edge('pipeline_comm_2_3', 'layer_3_attention_qkv')
    # ... (layer 3 processing)
    dot.edge('layer_3_final_layernorm', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_proposed_dag()
    dag.render('/home/wzc/data/file-share/2025-09-09-13-57-56/proposed_moe_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-09-13-57-56/proposed_moe_dag.dot')
    print("Proposed DAG generated successfully")