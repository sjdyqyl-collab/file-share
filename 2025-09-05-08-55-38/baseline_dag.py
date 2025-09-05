#!/usr/bin/env python3

import graphviz

def create_baseline_dag():
    """
    Create baseline DAG for 16 GPUs with:
    - TP=8, PP=2
    - 4 experts per GPU
    - 4-layer MoE model
    """
    dot = graphviz.Digraph('baseline_moe', 
                          comment='Baseline MoE Deployment (16 GPUs, TP=8, PP=2, 4 experts/GPU)')
    dot.attr(rankdir='TB', size='20,20')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')  # Input/Output
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightgreen')  # Computation
    dot.attr('node', shape='parallelogram', style='filled', fillcolor='lightyellow')  # Routing/Aggregation
    
    # Model dimensions
    batch_size = 1024
    seq_len = 512
    hidden_size = 8192  # 16 heads × 512
    expert_hidden = 32768
    num_experts = 16
    num_layers = 4
    
    # GPU assignments for baseline
    # PP stage 0: GPUs 0-7 (layers 0,1)
    # PP stage 1: GPUs 8-15 (layers 2,3)
    # Within each stage: TP=8 across GPUs
    # Each GPU hosts 4 experts (16 experts / 4 GPUs = 4 experts per GPU)
    
    # Input
    dot.node('input', f'Input\n[B×S×H]\n[{batch_size}×{seq_len}×{hidden_size}]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Layer 0 - Pipeline Stage 0
    with dot.subgraph(name='cluster_layer0') as c:
        c.attr(label='Layer 0 - Pipeline Stage 0 (GPUs 0-7)', style='dashed')
        
        # MHA across TP=8
        c.node('l0_mha_qkv', f'MHA QKV Linear\n[B×S×H]→[B×S×3H]\nSplit across 8 GPUs', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('l0_mha_split', f'Split Heads\n[B×S×3H]→[B×S×16×512×3]\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
        c.node('l0_mha_attn', f'Multi-Head Attention\n[B×S×16×512]\nTP=8 across GPUs 0-7', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('l0_mha_out', f'MHA Output Linear\n[B×S×H]→[B×S×H]\nTP=8 across GPUs 0-7', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('l0_mha_res', f'Residual Add\n[B×S×H] + [B×S×H]\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
        
        # Expert routing
        c.node('l0_gate', f'Expert Gate\n[B×S×H]→[B×S×16]\nRouting decisions\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
        
        # Expert computation - 4 experts per GPU
        for gpu_id in range(8):
            for expert_id in range(4):
                expert_idx = gpu_id * 4 + expert_id
                c.node(f'l0_exp_{gpu_id}_{expert_id}', 
                       f'Expert {expert_idx}\n[B×S×H]→[B×S×{expert_hidden}]→[B×S×H]\nGPU {gpu_id}', 
                       shape='rectangle', fillcolor='lightgreen')
        
        # Expert aggregation
        c.node('l0_exp_agg', f'Expert Aggregation\nSelect top-2 experts\nSum weighted outputs\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
        c.node('l0_exp_res', f'Residual Add\n[B×S×H] + [B×S×H]\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
    
    # Layer 1 - Pipeline Stage 0
    with dot.subgraph(name='cluster_layer1') as c:
        c.attr(label='Layer 1 - Pipeline Stage 0 (GPUs 0-7)', style='dashed')
        
        c.node('l1_mha_qkv', f'MHA QKV Linear\n[B×S×H]→[B×S×3H]\nSplit across 8 GPUs', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('l1_mha_split', f'Split Heads\n[B×S×3H]→[B×S×16×512×3]\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
        c.node('l1_mha_attn', f'Multi-Head Attention\n[B×S×16×512]\nTP=8 across GPUs 0-7', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('l1_mha_out', f'MHA Output Linear\n[B×S×H]→[B×S×H]\nTP=8 across GPUs 0-7', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('l1_mha_res', f'Residual Add\n[B×S×H] + [B×S×H]\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
        
        c.node('l1_gate', f'Expert Gate\n[B×S×H]→[B×S×16]\nRouting decisions\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
        
        for gpu_id in range(8):
            for expert_id in range(4):
                expert_idx = gpu_id * 4 + expert_id
                c.node(f'l1_exp_{gpu_id}_{expert_id}', 
                       f'Expert {expert_idx}\n[B×S×H]→[B×S×{expert_hidden}]→[B×S×H]\nGPU {gpu_id}', 
                       shape='rectangle', fillcolor='lightgreen')
        
        c.node('l1_exp_agg', f'Expert Aggregation\nSelect top-2 experts\nSum weighted outputs\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
        c.node('l1_exp_res', f'Residual Add\n[B×S×H] + [B×S×H]\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
    
    # Pipeline communication between stages
    dot.node('pipe_0_1', f'Pipeline Communication\nLayer 1→Layer 2\nGPUs 0-7 → GPUs 8-15\n[B×S×H]', 
             shape='parallelogram', fillcolor='orange')
    
    # Layer 2 - Pipeline Stage 1
    with dot.subgraph(name='cluster_layer2') as c:
        c.attr(label='Layer 2 - Pipeline Stage 1 (GPUs 8-15)', style='dashed')
        
        c.node('l2_mha_qkv', f'MHA QKV Linear\n[B×S×H]→[B×S×3H]\nSplit across 8 GPUs', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('l2_mha_split', f'Split Heads\n[B×S×3H]→[B×S×16×512×3]\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
        c.node('l2_mha_attn', f'Multi-Head Attention\n[B×S×16×512]\nTP=8 across GPUs 8-15', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('l2_mha_out', f'MHA Output Linear\n[B×S×H]→[B×S×H]\nTP=8 across GPUs 8-15', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('l2_mha_res', f'Residual Add\n[B×S×H] + [B×S×H]\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
        
        c.node('l2_gate', f'Expert Gate\n[B×S×H]→[B×S×16]\nRouting decisions\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
        
        for gpu_id in range(8, 16):
            for expert_id in range(4):
                expert_idx = (gpu_id - 8) * 4 + expert_id
                c.node(f'l2_exp_{gpu_id}_{expert_id}', 
                       f'Expert {expert_idx}\n[B×S×H]→[B×S×{expert_hidden}]→[B×S×H]\nGPU {gpu_id}', 
                       shape='rectangle', fillcolor='lightgreen')
        
        c.node('l2_exp_agg', f'Expert Aggregation\nSelect top-2 experts\nSum weighted outputs\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
        c.node('l2_exp_res', f'Residual Add\n[B×S×H] + [B×S×H]\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
    
    # Layer 3 - Pipeline Stage 1
    with dot.subgraph(name='cluster_layer3') as c:
        c.attr(label='Layer 3 - Pipeline Stage 1 (GPUs 8-15)', style='dashed')
        
        c.node('l3_mha_qkv', f'MHA QKV Linear\n[B×S×H]→[B×S×3H]\nSplit across 8 GPUs', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('l3_mha_split', f'Split Heads\n[B×S×3H]→[B×S×16×512×3]\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
        c.node('l3_mha_attn', f'Multi-Head Attention\n[B×S×16×512]\nTP=8 across GPUs 8-15', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('l3_mha_out', f'MHA Output Linear\n[B×S×H]→[B×S×H]\nTP=8 across GPUs 8-15', 
               shape='rectangle', fillcolor='lightgreen')
        c.node('l3_mha_res', f'Residual Add\n[B×S×H] + [B×S×H]\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
        
        c.node('l3_gate', f'Expert Gate\n[B×S×H]→[B×S×16]\nRouting decisions\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
        
        for gpu_id in range(8, 16):
            for expert_id in range(4):
                expert_idx = (gpu_id - 8) * 4 + expert_id
                c.node(f'l3_exp_{gpu_id}_{expert_id}', 
                       f'Expert {expert_idx}\n[B×S×H]→[B×S×{expert_hidden}]→[B×S×H]\nGPU {gpu_id}', 
                       shape='rectangle', fillcolor='lightgreen')
        
        c.node('l3_exp_agg', f'Expert Aggregation\nSelect top-2 experts\nSum weighted outputs\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
        c.node('l3_exp_res', f'Residual Add\n[B×S×H] + [B×S×H]\nAll GPUs', 
               shape='parallelogram', fillcolor='lightyellow')
    
    # Output
    dot.node('output', f'Output\n[B×S×H]\n[{batch_size}×{seq_len}×{hidden_size}]', 
             shape='ellipse', fillcolor='lightblue')
    
    # Connections
    dot.edge('input', 'l0_mha_qkv')
    dot.edge('l0_mha_qkv', 'l0_mha_split')
    dot.edge('l0_mha_split', 'l0_mha_attn')
    dot.edge('l0_mha_attn', 'l0_mha_out')
    dot.edge('l0_mha_out', 'l0_mha_res')
    dot.edge('input', 'l0_mha_res')  # Residual connection
    dot.edge('l0_mha_res', 'l0_gate')
    
    # Connect experts for layer 0
    for gpu_id in range(8):
        for expert_id in range(4):
            dot.edge('l0_gate', f'l0_exp_{gpu_id}_{expert_id}', style='dashed')
            dot.edge(f'l0_exp_{gpu_id}_{expert_id}', 'l0_exp_agg')
    
    dot.edge('l0_exp_agg', 'l0_exp_res')
    dot.edge('l0_mha_res', 'l0_exp_res')  # Residual connection
    dot.edge('l0_exp_res', 'l1_mha_qkv')
    
    # Layer 1 connections
    dot.edge('l1_mha_qkv', 'l1_mha_split')
    dot.edge('l1_mha_split', 'l1_mha_attn')
    dot.edge('l1_mha_attn', 'l1_mha_out')
    dot.edge('l1_mha_out', 'l1_mha_res')
    dot.edge('l0_exp_res', 'l1_mha_res')  # Residual connection
    dot.edge('l1_mha_res', 'l1_gate')
    
    for gpu_id in range(8):
        for expert_id in range(4):
            dot.edge('l1_gate', f'l1_exp_{gpu_id}_{expert_id}', style='dashed')
            dot.edge(f'l1_exp_{gpu_id}_{expert_id}', 'l1_exp_agg')
    
    dot.edge('l1_exp_agg', 'l1_exp_res')
    dot.edge('l1_mha_res', 'l1_exp_res')  # Residual connection
    dot.edge('l1_exp_res', 'pipe_0_1')
    
    # Pipeline stage 1
    dot.edge('pipe_0_1', 'l2_mha_qkv')
    dot.edge('l2_mha_qkv', 'l2_mha_split')
    dot.edge('l2_mha_split', 'l2_mha_attn')
    dot.edge('l2_mha_attn', 'l2_mha_out')
    dot.edge('l2_mha_out', 'l2_mha_res')
    dot.edge('pipe_0_1', 'l2_mha_res')  # Residual connection
    dot.edge('l2_mha_res', 'l2_gate')
    
    for gpu_id in range(8, 16):
        for expert_id in range(4):
            dot.edge('l2_gate', f'l2_exp_{gpu_id}_{expert_id}', style='dashed')
            dot.edge(f'l2_exp_{gpu_id}_{expert_id}', 'l2_exp_agg')
    
    dot.edge('l2_exp_agg', 'l2_exp_res')
    dot.edge('l2_mha_res', 'l2_exp_res')  # Residual connection
    dot.edge('l2_exp_res', 'l3_mha_qkv')
    
    # Layer 3 connections
    dot.edge('l3_mha_qkv', 'l3_mha_split')
    dot.edge('l3_mha_split', 'l3_mha_attn')
    dot.edge('l3_mha_attn', 'l3_mha_out')
    dot.edge('l3_mha_out', 'l3_mha_res')
    dot.edge('l2_exp_res', 'l3_mha_res')  # Residual connection
    dot.edge('l3_mha_res', 'l3_gate')
    
    for gpu_id in range(8, 16):
        for expert_id in range(4):
            dot.edge('l3_gate', f'l3_exp_{gpu_id}_{expert_id}', style='dashed')
            dot.edge(f'l3_exp_{gpu_id}_{expert_id}', 'l3_exp_agg')
    
    dot.edge('l3_exp_agg', 'l3_exp_res')
    dot.edge('l3_mha_res', 'l3_exp_res')  # Residual connection
    dot.edge('l3_exp_res', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/2025-09-05-08-55-38/baseline_moe', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-05-08-55-38/baseline_moe.dot')
    print("Baseline DAG generated successfully")