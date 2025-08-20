#!/usr/bin/env python3

import graphviz

def create_baseline_dag():
    """
    Create DAG for baseline deployment: 16 GPUs, TP=8, PP=2, 4 experts per GPU
    """
    dot = graphviz.Digraph(comment='Baseline MoE Deployment')
    dot.attr(rankdir='TB', size='20,30')
    
    # Global attributes
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Input layer
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input (Batch: 1024 tokens)', style='rounded')
        c.node('input', 'Input Tokens\n[1024, 8192]', shape='ellipse')
    
    # Pipeline Stage 0 (Layers 0-1)
    with dot.subgraph(name='cluster_pipeline0') as c:
        c.attr(label='Pipeline Stage 0 (Layers 0-1)\nGPUs 0-7', style='rounded', color='blue')
        
        # Layer 0
        with c.subgraph(name='cluster_layer0') as layer:
            layer.attr(label='Layer 0', style='rounded')
            
            # MHA across 8 GPUs (TP=8)
            layer.node('l0_mha_qkv', 'MHA QKV Linear\n[1024, 8192] → [1024, 3×16×512]\nTP=8 across GPUs 0-7', shape='rectangle')
            layer.node('l0_mha_attn', 'MHA Attention\n[1024, 16×512]\nTP=8 across GPUs 0-7', shape='rectangle')
            layer.node('l0_mha_out', 'MHA Output Linear\n[1024, 8192] → [1024, 8192]\nTP=8 across GPUs 0-7', shape='rectangle')
            layer.node('l0_mha_res', 'Residual Add\n[1024, 8192] + [1024, 8192]\nAll GPUs 0-7', shape='parallelogram')
            
            # MoE Layer - 16 experts distributed across 8 GPUs (2 experts per GPU)
            for gpu_id in range(8):
                with layer.subgraph(name=f'cluster_gpu{gpu_id}') as gpu:
                    gpu.attr(label=f'GPU {gpu_id}', style='dashed')
                    
                    # Gate computation
                    gpu.node(f'l0_gate_{gpu_id}', f'Gate\n[1024, 8192] → [1024, 16]\nGPU {gpu_id}', shape='parallelogram')
                    
                    # 2 experts per GPU
                    for expert_idx in range(2):
                        expert_id = gpu_id * 2 + expert_idx
                        gpu.node(f'l0_exp{expert_id}', f'Expert {expert_id}\nMLP\n[1024, 8192] → [1024, 32768] → [1024, 8192]\nGPU {gpu_id}', shape='rectangle')
                    
                    # Expert aggregation
                    gpu.node(f'l0_agg_{gpu_id}', f'Expert Aggregation\n[1024, 8192]\nGPU {gpu_id}', shape='parallelogram')
            
            layer.node('l0_moe_res', 'Residual Add\n[1024, 8192] + [1024, 8192]\nAll GPUs 0-7', shape='parallelogram')
    
    # Pipeline Stage 1 (Layers 2-3)
    with dot.subgraph(name='cluster_pipeline1') as c:
        c.attr(label='Pipeline Stage 1 (Layers 2-3)\nGPUs 8-15', style='rounded', color='red')
        
        # Similar structure for layers 2-3
        for layer_id in [2, 3]:
            with c.subgraph(name=f'cluster_layer{layer_id}') as layer:
                layer.attr(label=f'Layer {layer_id}', style='rounded')
                
                # MHA
                layer.node(f'l{layer_id}_mha_qkv', f'MHA QKV Linear\n[1024, 8192] → [1024, 3×16×512]\nTP=8 across GPUs {(layer_id*8)-16}-{(layer_id*8)-9}', shape='rectangle')
                layer.node(f'l{layer_id}_mha_attn', f'MHA Attention\n[1024, 16×512]\nTP=8 across GPUs {(layer_id*8)-16}-{(layer_id*8)-9}', shape='rectangle')
                layer.node(f'l{layer_id}_mha_out', f'MHA Output Linear\n[1024, 8192] → [1024, 8192]\nTP=8 across GPUs {(layer_id*8)-16}-{(layer_id*8)-9}', shape='rectangle')
                layer.node(f'l{layer_id}_mha_res', f'Residual Add\n[1024, 8192] + [1024, 8192]\nAll GPUs {(layer_id*8)-16}-{(layer_id*8)-9}', shape='parallelogram')
                
                # MoE
                for gpu_id in range(8):
                    actual_gpu = layer_id * 8 + gpu_id
                    with layer.subgraph(name=f'cluster_gpu{actual_gpu}') as gpu:
                        gpu.attr(label=f'GPU {actual_gpu}', style='dashed')
                        
                        gpu.node(f'l{layer_id}_gate_{actual_gpu}', f'Gate\n[1024, 8192] → [1024, 16]\nGPU {actual_gpu}', shape='parallelogram')
                        
                        for expert_idx in range(2):
                            expert_id = gpu_id * 2 + expert_idx
                            gpu.node(f'l{layer_id}_exp{expert_id}', f'Expert {expert_id}\nMLP\n[1024, 8192] → [1024, 32768] → [1024, 8192]\nGPU {actual_gpu}', shape='rectangle')
                        
                        gpu.node(f'l{layer_id}_agg_{actual_gpu}', f'Expert Aggregation\n[1024, 8192]\nGPU {actual_gpu}', shape='parallelogram')
                
                layer.node(f'l{layer_id}_moe_res', f'Residual Add\n[1024, 8192] + [1024, 8192]\nAll GPUs {(layer_id*8)-16}-{(layer_id*8)-9}', shape='parallelogram')
    
    # Output layer
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output', style='rounded')
        c.node('output', 'Output Tokens\n[1024, 8192]', shape='ellipse')
    
    # Connections
    # Input to Layer 0 MHA
    dot.edge('input', 'l0_mha_qkv')
    dot.edge('l0_mha_qkv', 'l0_mha_attn')
    dot.edge('l0_mha_attn', 'l0_mha_out')
    dot.edge('l0_mha_out', 'l0_mha_res')
    dot.edge('input', 'l0_mha_res')  # Residual connection
    
    # Layer 0 MHA to MoE
    dot.edge('l0_mha_res', 'l0_gate_0')
    for gpu_id in range(8):
        dot.edge('l0_mha_res', f'l0_gate_{gpu_id}')
        for expert_idx in range(2):
            expert_id = gpu_id * 2 + expert_idx
            dot.edge(f'l0_gate_{gpu_id}', f'l0_exp{expert_id}', style='dashed', label=f'route')
            dot.edge(f'l0_exp{expert_id}', f'l0_agg_{gpu_id}')
        dot.edge(f'l0_agg_{gpu_id}', 'l0_moe_res')
    dot.edge('l0_mha_res', 'l0_moe_res')  # Residual connection
    
    # Pipeline communication
    dot.edge('l0_moe_res', 'l2_mha_qkv', label='PP stage transfer\nGPUs 0-7 → 8-15')
    
    # Layer 2 connections
    dot.edge('l2_mha_qkv', 'l2_mha_attn')
    dot.edge('l2_mha_attn', 'l2_mha_out')
    dot.edge('l2_mha_out', 'l2_mha_res')
    dot.edge('l2_mha_qkv', 'l2_mha_res')  # Residual
    
    dot.edge('l2_mha_res', 'l2_gate_16')
    for gpu_id in range(8):
        actual_gpu = 16 + gpu_id
        dot.edge('l2_mha_res', f'l2_gate_{actual_gpu}')
        for expert_idx in range(2):
            expert_id = gpu_id * 2 + expert_idx
            dot.edge(f'l2_gate_{actual_gpu}', f'l2_exp{expert_id}', style='dashed', label=f'route')
            dot.edge(f'l2_exp{expert_id}', f'l2_agg_{actual_gpu}')
        dot.edge(f'l2_agg_{actual_gpu}', 'l2_moe_res')
    dot.edge('l2_mha_res', 'l2_moe_res')  # Residual
    
    dot.edge('l2_moe_res', 'l3_mha_qkv')
    
    # Layer 3 connections
    dot.edge('l3_mha_qkv', 'l3_mha_attn')
    dot.edge('l3_mha_attn', 'l3_mha_out')
    dot.edge('l3_mha_out', 'l3_mha_res')
    dot.edge('l3_mha_qkv', 'l3_mha_res')  # Residual
    
    dot.edge('l3_mha_res', 'l3_gate_24')
    for gpu_id in range(8):
        actual_gpu = 24 + gpu_id
        dot.edge('l3_mha_res', f'l3_gate_{actual_gpu}')
        for expert_idx in range(2):
            expert_id = gpu_id * 2 + expert_idx
            dot.edge(f'l3_gate_{actual_gpu}', f'l3_exp{expert_id}', style='dashed', label=f'route')
            dot.edge(f'l3_exp{expert_id}', f'l3_agg_{actual_gpu}')
        dot.edge(f'l3_agg_{actual_gpu}', 'l3_moe_res')
    dot.edge('l3_mha_res', 'l3_moe_res')  # Residual
    
    dot.edge('l3_moe_res', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/submission/baseline_moe_deployment', format='svg', cleanup=True)
    print("Baseline DAG generated successfully")