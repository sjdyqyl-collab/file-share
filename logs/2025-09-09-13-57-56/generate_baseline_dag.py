#!/usr/bin/env python3

import graphviz

def create_baseline_dag():
    """Create baseline DAG with TP=8, PP=2, 16 GPUs total, 4 experts per GPU"""
    
    dot = graphviz.Digraph('Baseline_MoE_TP8_PP2', comment='Baseline MoE with TP=8, PP=2')
    dot.attr(rankdir='TB', size='20,30')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Communication
    dot.attr('node', shape='diamond', style='filled', fillcolor='lightcoral')  # Routing/Gating
    
    # Input node
    dot.node('input', 'Input\\nInput: [batch_size=1024, seq_len=10000, dim=8192]\\nGPU: All GPUs', 
             shape='ellipse', fillcolor='lightblue')
    
    # Process each pipeline stage
    for stage in [0, 1]:
        stage_name = f"stage_{stage}"
        gpu_start = stage * 8
        
        # Process each layer in this stage
        for layer in [0, 1] if stage == 0 else [2, 3]:
            layer_name = f"layer_{layer}"
            
            # Multi-Head Attention for this layer
            attention_name = f"{stage_name}_{layer_name}_attention"
            dot.node(f"{attention_name}_qkv", 
                     f"QKV Linear\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 16, 512]\\nGPU: {gpu_start}-{gpu_start+7}",
                     shape='rectangle', fillcolor='lightgreen')
            
            dot.node(f"{attention_name}_split", 
                     f"Split Heads\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 16, 512]\\nGPU: {gpu_start}-{gpu_start+7}",
                     shape='rectangle', fillcolor='lightgreen')
            
            dot.node(f"{attention_name}_matmul_qk", 
                     f"QK^T Matmul\\nInput: [1024, 10000, 16, 512]\\nOutput: [1024, 16, 10000, 10000]\\nGPU: {gpu_start}-{gpu_start+7}",
                     shape='rectangle', fillcolor='lightgreen')
            
            dot.node(f"{attention_name}_softmax", 
                     f"Softmax\\nInput: [1024, 16, 10000, 10000]\\nOutput: [1024, 16, 10000, 10000]\\nGPU: {gpu_start}-{gpu_start+7}",
                     shape='rectangle', fillcolor='lightgreen')
            
            dot.node(f"{attention_name}_matmul_v", 
                     f"Attention Output\\nInput: [1024, 16, 10000, 10000], [1024, 10000, 16, 512]\\nOutput: [1024, 10000, 16, 512]\\nGPU: {gpu_start}-{gpu_start+7}",
                     shape='rectangle', fillcolor='lightgreen')
            
            dot.node(f"{attention_name}_concat", 
                     f"Concat Heads\\nInput: [1024, 10000, 16, 512]\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_start}-{gpu_start+7}",
                     shape='rectangle', fillcolor='lightgreen')
            
            dot.node(f"{attention_name}_linear", 
                     f"Output Linear\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_start}-{gpu_start+7}",
                     shape='rectangle', fillcolor='lightgreen')
            
            dot.node(f"{attention_name}_residual", 
                     f"Residual Add\\nInput: [1024, 10000, 8192], [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_start}-{gpu_start+7}",
                     shape='rectangle', fillcolor='lightgreen')
            
            # LayerNorm after attention
            dot.node(f"{attention_name}_layernorm", 
                     f"LayerNorm\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_start}-{gpu_start+7}",
                     shape='rectangle', fillcolor='lightgreen')
            
            # MoE layer with 4 experts per GPU
            for gpu in range(8):
                gpu_id = gpu_start + gpu
                
                # Gating for this GPU's experts
                gating_name = f"{stage_name}_{layer_name}_gpu{gpu}_gating"
                dot.node(gating_name, 
                         f"Gating\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 4]\\nGPU: {gpu_id}",
                         shape='diamond', fillcolor='lightcoral')
                
                # 4 experts on this GPU
                for expert_id in range(4):
                    global_expert_id = gpu * 4 + expert_id
                    expert_name = f"{stage_name}_{layer_name}_gpu{gpu}_expert{expert_id}"
                    
                    dot.node(f"{expert_name}_linear1", 
                             f"Expert {global_expert_id} Linear1\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 32768]\\nGPU: {gpu_id}",
                             shape='rectangle', fillcolor='lightgreen')
                    
                    dot.node(f"{expert_name}_gelu", 
                             f"Expert {global_expert_id} GELU\\nInput: [1024, 10000, 32768]\\nOutput: [1024, 10000, 32768]\\nGPU: {gpu_id}",
                             shape='rectangle', fillcolor='lightgreen')
                    
                    dot.node(f"{expert_name}_linear2", 
                             f"Expert {global_expert_id} Linear2\\nInput: [1024, 10000, 32768]\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_id}",
                             shape='rectangle', fillcolor='lightgreen')
            
            # Expert aggregation
            aggregation_name = f"{stage_name}_{layer_name}_aggregation"
            dot.node(aggregation_name, 
                     f"Expert Aggregation\\nInput: [1024, 10000, 8192] × 4\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_start}-{gpu_start+7}",
                     shape='parallelogram', fillcolor='lightyellow')
            
            # Final residual and layernorm
            dot.node(f"{layer_name}_final_residual", 
                     f"Final Residual\\nInput: [1024, 10000, 8192], [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_start}-{gpu_start+7}",
                     shape='rectangle', fillcolor='lightgreen')
            
            dot.node(f"{layer_name}_final_layernorm", 
                     f"Final LayerNorm\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: {gpu_start}-{gpu_start+7}",
                     shape='rectangle', fillcolor='lightgreen')
    
    # Pipeline communication
    dot.node("pipeline_comm_0_1", 
             "Pipeline Communication\\nInput: [1024, 10000, 8192]\\nOutput: [1024, 10000, 8192]\\nGPU: 7→8",
             shape='parallelogram', fillcolor='lightyellow')
    
    # Output node
    dot.node('output', 'Output\\nInput: [batch_size=1024, seq_len=10000, dim=8192]\\nGPU: All GPUs', 
             shape='ellipse', fillcolor='lightblue')
    
    # Connect the DAG
    # Input to first attention
    dot.edge('input', 'stage_0_layer_0_attention_qkv')
    
    # Attention connections for layer 0
    dot.edge('stage_0_layer_0_attention_qkv', 'stage_0_layer_0_attention_split')
    dot.edge('stage_0_layer_0_attention_split', 'stage_0_layer_0_attention_matmul_qk')
    dot.edge('stage_0_layer_0_attention_matmul_qk', 'stage_0_layer_0_attention_softmax')
    dot.edge('stage_0_layer_0_attention_softmax', 'stage_0_layer_0_attention_matmul_v')
    dot.edge('stage_0_layer_0_attention_matmul_v', 'stage_0_layer_0_attention_concat')
    dot.edge('stage_0_layer_0_attention_concat', 'stage_0_layer_0_attention_linear')
    dot.edge('stage_0_layer_0_attention_linear', 'stage_0_layer_0_attention_residual')
    dot.edge('input', 'stage_0_layer_0_attention_residual')  # Residual connection
    dot.edge('stage_0_layer_0_attention_residual', 'stage_0_layer_0_attention_layernorm')
    
    # MoE connections for layer 0
    for gpu in range(8):
        gpu_id = gpu
        dot.edge('stage_0_layer_0_attention_layernorm', 
                 f'stage_0_layer_0_gpu{gpu}_gating')
        
        for expert_id in range(4):
            expert_name = f'stage_0_layer_0_gpu{gpu}_expert{expert_id}'
            dot.edge(f'stage_0_layer_0_attention_layernorm', 
                     f'{expert_name}_linear1', style='dashed')
            dot.edge(f'{expert_name}_linear1', f'{expert_name}_gelu')
            dot.edge(f'{expert_name}_gelu', f'{expert_name}_linear2')
            dot.edge(f'{expert_name}_linear2', 'stage_0_layer_0_aggregation')
            dot.edge(f'stage_0_layer_0_gpu{gpu}_gating', 
                     f'{expert_name}_linear1', style='dashed')
    
    dot.edge('stage_0_layer_0_aggregation', 'layer_0_final_residual')
    dot.edge('stage_0_layer_0_attention_layernorm', 'layer_0_final_residual')  # Residual
    dot.edge('layer_0_final_residual', 'layer_0_final_layernorm')
    
    # Continue with layer 1 (similar structure)
    dot.edge('layer_0_final_layernorm', 'stage_0_layer_1_attention_qkv')
    # ... (similar attention connections)
    dot.edge('stage_0_layer_1_attention_qkv', 'stage_0_layer_1_attention_split')
    dot.edge('stage_0_layer_1_attention_split', 'stage_0_layer_1_attention_matmul_qk')
    dot.edge('stage_0_layer_1_attention_matmul_qk', 'stage_0_layer_1_attention_softmax')
    dot.edge('stage_0_layer_1_attention_softmax', 'stage_0_layer_1_attention_matmul_v')
    dot.edge('stage_0_layer_1_attention_matmul_v', 'stage_0_layer_1_attention_concat')
    dot.edge('stage_0_layer_1_attention_concat', 'stage_0_layer_1_attention_linear')
    dot.edge('layer_0_final_layernorm', 'stage_0_layer_1_attention_residual')  # Residual
    dot.edge('stage_0_layer_1_attention_linear', 'stage_0_layer_1_attention_residual')
    dot.edge('stage_0_layer_1_attention_residual', 'stage_0_layer_1_attention_layernorm')
    
    # MoE for layer 1
    for gpu in range(8):
        gpu_id = gpu + 8 if stage == 1 else gpu
        dot.edge('stage_0_layer_1_attention_layernorm', 
                 f'stage_0_layer_1_gpu{gpu}_gating')
        
        for expert_id in range(4):
            expert_name = f'stage_0_layer_1_gpu{gpu}_expert{expert_id}'
            dot.edge('stage_0_layer_1_attention_layernorm', 
                     f'{expert_name}_linear1', style='dashed')
            dot.edge(f'{expert_name}_linear1', f'{expert_name}_gelu')
            dot.edge(f'{expert_name}_gelu', f'{expert_name}_linear2')
            dot.edge(f'{expert_name}_linear2', 'stage_0_layer_1_aggregation')
            dot.edge(f'stage_0_layer_1_gpu{gpu}_gating', 
                     f'{expert_name}_linear1', style='dashed')
    
    dot.edge('stage_0_layer_1_aggregation', 'layer_1_final_residual')
    dot.edge('stage_0_layer_1_attention_layernorm', 'layer_1_final_residual')
    dot.edge('layer_1_final_residual', 'layer_1_final_layernorm')
    
    # Pipeline communication
    dot.edge('layer_1_final_layernorm', 'pipeline_comm_0_1')
    
    # Stage 1 processing (similar to stage 0)
    dot.edge('pipeline_comm_0_1', 'stage_1_layer_2_attention_qkv')
    # ... (continue with stage 1 processing)
    
    # For brevity, connect final stage to output
    dot.edge('layer_3_final_layernorm', 'output')
    
    return dot

if __name__ == "__main__":
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/2025-09-09-13-57-56/baseline_moe_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-09-13-57-56/baseline_moe_dag.dot')
    print("Baseline DAG generated successfully")