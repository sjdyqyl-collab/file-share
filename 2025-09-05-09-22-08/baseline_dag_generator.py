#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_baseline_dag():
    """Create baseline DAG with TP=8, PP=2, 16 GPUs, 4 experts per GPU"""
    
    dot = Digraph(comment='Baseline MoE Deployment: TP=8, PP=2')
    dot.attr(rankdir='TB', size='20,30', fontname='Arial')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fontname='Arial')
    dot.attr('edge', fontname='Arial', fontsize='10')
    
    # Colors for different GPU groups
    colors = {
        'stage0': 'lightblue',
        'stage1': 'lightgreen',
        'communication': 'yellow',
        'input': 'lightgrey',
        'output': 'lightcoral'
    }
    
    # Global input
    dot.node('input', 'Input\n[1024, 2048, 8192]', fillcolor=colors['input'])
    
    # Stage 0 (GPUs 0-7)
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0 (GPUs 0-7)', style='dashed', color='blue')
        
        # Layer 0 - Attention across 8 GPUs with TP=8
        stage0.node('stage0_attn_qkv', 'Attention QKV\n[1024, 8192] -> [1024, 16, 512]\nTP=8 shard\nGPU: 0-7', fillcolor=colors['stage0'])
        stage0.node('stage0_attn_score', 'Attention Score\n[1024, 16, 512] x [512, 2048]\nTP=8 shard\nGPU: 0-7', fillcolor=colors['stage0'])
        stage0.node('stage0_attn_out', 'Attention Output\n[1024, 16, 512] -> [1024, 8192]\nTP=8 shard\nGPU: 0-7', fillcolor=colors['stage0'])
        stage0.node('stage0_attn_residual', 'Residual Add\n[1024, 8192] + [1024, 8192]\nGPU: 0-7', fillcolor=colors['stage0'])
        
        # Layer 0 - MoE with 16 experts distributed across 8 GPUs (2 per GPU)
        stage0.node('stage0_gate', 'Gate Network\n[1024, 8192] -> [1024, 16]\nGPU: 0-7', fillcolor=colors['stage0'])
        
        # Experts on each GPU (2 experts per GPU, 8 GPUs = 16 experts total)
        for gpu_id in range(8):
            stage0.node(f'stage0_expert0_gpu{gpu_id}', f'Expert {gpu_id*2}\n[1024, 8192] -> [1024, 32768] -> [1024, 8192]\nTP=8 shard\nGPU: {gpu_id}', fillcolor=colors['stage0'])
            stage0.node(f'stage0_expert1_gpu{gpu_id}', f'Expert {gpu_id*2+1}\n[1024, 8192] -> [1024, 32768] -> [1024, 8192]\nTP=8 shard\nGPU: {gpu_id}', fillcolor=colors['stage0'])
        
        stage0.node('stage0_moe_aggregate', 'Expert Aggregation\n[1024, 8192] x k=2\nGPU: 0-7', fillcolor=colors['stage0'])
        stage0.node('stage0_moe_residual', 'Residual Add\n[1024, 8192] + [1024, 8192]\nGPU: 0-7', fillcolor=colors['stage0'])
    
    # Communication between stages
    dot.node('comm_stage0_to_stage1', 'Pipeline Communication\n[1024, 8192]\nGPU: 7 -> GPU: 8', fillcolor=colors['communication'], shape='ellipse')
    
    # Stage 1 (GPUs 8-15)
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1 (GPUs 8-15)', style='dashed', color='green')
        
        # Layer 1 - Attention across 8 GPUs with TP=8
        stage1.node('stage1_attn_qkv', 'Attention QKV\n[1024, 8192] -> [1024, 16, 512]\nTP=8 shard\nGPU: 8-15', fillcolor=colors['stage1'])
        stage1.node('stage1_attn_score', 'Attention Score\n[1024, 16, 512] x [512, 2048]\nTP=8 shard\nGPU: 8-15', fillcolor=colors['stage1'])
        stage1.node('stage1_attn_out', 'Attention Output\n[1024, 16, 512] -> [1024, 8192]\nTP=8 shard\nGPU: 8-15', fillcolor=colors['stage1'])
        stage1.node('stage1_attn_residual', 'Residual Add\n[1024, 8192] + [1024, 8192]\nGPU: 8-15', fillcolor=colors['stage1'])
        
        # Layer 1 - MoE with 16 experts distributed across 8 GPUs
        stage1.node('stage1_gate', 'Gate Network\n[1024, 8192] -> [1024, 16]\nGPU: 8-15', fillcolor=colors['stage1'])
        
        # Experts on each GPU (2 experts per GPU, 8 GPUs = 16 experts total)
        for gpu_id in range(8, 16):
            expert_base = (gpu_id - 8) * 2 + 16
            stage1.node(f'stage1_expert0_gpu{gpu_id}', f'Expert {expert_base}\n[1024, 8192] -> [1024, 32768] -> [1024, 8192]\nTP=8 shard\nGPU: {gpu_id}', fillcolor=colors['stage1'])
            stage1.node(f'stage1_expert1_gpu{gpu_id}', f'Expert {expert_base+1}\n[1024, 8192] -> [1024, 32768] -> [1024, 8192]\nTP=8 shard\nGPU: {gpu_id}', fillcolor=colors['stage1'])
        
        stage1.node('stage1_moe_aggregate', 'Expert Aggregation\n[1024, 8192] x k=2\nGPU: 8-15', fillcolor=colors['stage1'])
        stage1.node('stage1_moe_residual', 'Residual Add\n[1024, 8192] + [1024, 8192]\nGPU: 8-15', fillcolor=colors['stage1'])
    
    # Layer 2 - Back to Stage 0 (pipeline loop)
    dot.node('comm_stage1_to_stage0', 'Pipeline Communication\n[1024, 8192]\nGPU: 15 -> GPU: 0', fillcolor=colors['communication'], shape='ellipse')
    
    # Layer 2 - Stage 0
    stage0.node('stage2_attn_qkv', 'Attention QKV\n[1024, 8192] -> [1024, 16, 512]\nTP=8 shard\nGPU: 0-7', fillcolor=colors['stage0'])
    stage0.node('stage2_attn_score', 'Attention Score\n[1024, 16, 512] x [512, 2048]\nTP=8 shard\nGPU: 0-7', fillcolor=colors['stage0'])
    stage0.node('stage2_attn_out', 'Attention Output\n[1024, 16, 512] -> [1024, 8192]\nTP=8 shard\nGPU: 0-7', fillcolor=colors['stage0'])
    stage0.node('stage2_attn_residual', 'Residual Add\n[1024, 8192] + [1024, 8192]\nGPU: 0-7', fillcolor=colors['stage0'])
    
    stage0.node('stage2_gate', 'Gate Network\n[1024, 8192] -> [1024, 16]\nGPU: 0-7', fillcolor=colors['stage0'])
    
    for gpu_id in range(8):
        expert_base = gpu_id * 2 + 32
        stage0.node(f'stage2_expert0_gpu{gpu_id}', f'Expert {expert_base}\n[1024, 8192] -> [1024, 32768] -> [1024, 8192]\nTP=8 shard\nGPU: {gpu_id}', fillcolor=colors['stage0'])
        stage0.node(f'stage2_expert1_gpu{gpu_id}', f'Expert {expert_base+1}\n[1024, 8192] -> [1024, 32768] -> [1024, 8192]\nTP=8 shard\nGPU: {gpu_id}', fillcolor=colors['stage0'])
    
    stage0.node('stage2_moe_aggregate', 'Expert Aggregation\n[1024, 8192] x k=2\nGPU: 0-7', fillcolor=colors['stage0'])
    stage0.node('stage2_moe_residual', 'Residual Add\n[1024, 8192] + [1024, 8192]\nGPU: 0-7', fillcolor=colors['stage0'])
    
    # Layer 3 - Stage 1
    dot.node('comm_stage2_to_stage1_again', 'Pipeline Communication\n[1024, 8192]\nGPU: 7 -> GPU: 8', fillcolor=colors['communication'], shape='ellipse')
    
    stage1.node('stage3_attn_qkv', 'Attention QKV\n[1024, 8192] -> [1024, 16, 512]\nTP=8 shard\nGPU: 8-15', fillcolor=colors['stage1'])
    stage1.node('stage3_attn_score', 'Attention Score\n[1024, 16, 512] x [512, 2048]\nTP=8 shard\nGPU: 8-15', fillcolor=colors['stage1'])
    stage1.node('stage3_attn_out', 'Attention Output\n[1024, 16, 512] -> [1024, 8192]\nTP=8 shard\nGPU: 8-15', fillcolor=colors['stage1'])
    stage1.node('stage3_attn_residual', 'Residual Add\n[1024, 8192] + [1024, 8192]\nGPU: 8-15', fillcolor=colors['stage1'])
    
    stage1.node('stage3_gate', 'Gate Network\n[1024, 8192] -> [1024, 16]\nGPU: 8-15', fillcolor=colors['stage1'])
    
    for gpu_id in range(8, 16):
        expert_base = (gpu_id - 8) * 2 + 48
        stage1.node(f'stage3_expert0_gpu{gpu_id}', f'Expert {expert_base}\n[1024, 8192] -> [1024, 32768] -> [1024, 8192]\nTP=8 shard\nGPU: {gpu_id}', fillcolor=colors['stage1'])
        stage1.node(f'stage3_expert1_gpu{gpu_id}', f'Expert {expert_base+1}\n[1024, 8192] -> [1024, 32768] -> [1024, 8192]\nTP=8 shard\nGPU: {gpu_id}', fillcolor=colors['stage1'])
    
    stage1.node('stage3_moe_aggregate', 'Expert Aggregation\n[1024, 8192] x k=2\nGPU: 8-15', fillcolor=colors['stage1'])
    stage1.node('stage3_moe_residual', 'Residual Add\n[1024, 8192] + [1024, 8192]\nGPU: 8-15', fillcolor=colors['stage1'])
    
    # Global output
    dot.node('output', 'Output\n[1024, 2048, 8192]', fillcolor=colors['output'])
    
    # Connections
    # Input to first stage
    dot.edge('input', 'stage0_attn_qkv')
    
    # Layer 0 connections
    dot.edge('stage0_attn_qkv', 'stage0_attn_score')
    dot.edge('stage0_attn_score', 'stage0_attn_out')
    dot.edge('stage0_attn_out', 'stage0_attn_residual')
    dot.edge('stage0_attn_residual', 'stage0_gate')
    
    # Expert routing (dashed lines)
    for gpu_id in range(8):
        dot.edge('stage0_gate', f'stage0_expert0_gpu{gpu_id}', style='dashed', label='route')
        dot.edge('stage0_gate', f'stage0_expert1_gpu{gpu_id}', style='dashed', label='route')
        dot.edge(f'stage0_expert0_gpu{gpu_id}', 'stage0_moe_aggregate')
        dot.edge(f'stage0_expert1_gpu{gpu_id}', 'stage0_moe_aggregate')
    
    dot.edge('stage0_moe_aggregate', 'stage0_moe_residual')
    dot.edge('stage0_moe_residual', 'comm_stage0_to_stage1')
    
    # Pipeline communication
    dot.edge('comm_stage0_to_stage1', 'stage1_attn_qkv')
    
    # Layer 1 connections
    dot.edge('stage1_attn_qkv', 'stage1_attn_score')
    dot.edge('stage1_attn_score', 'stage1_attn_out')
    dot.edge('stage1_attn_out', 'stage1_attn_residual')
    dot.edge('stage1_attn_residual', 'stage1_gate')
    
    for gpu_id in range(8, 16):
        dot.edge('stage1_gate', f'stage1_expert0_gpu{gpu_id}', style='dashed', label='route')
        dot.edge('stage1_gate', f'stage1_expert1_gpu{gpu_id}', style='dashed', label='route')
        dot.edge(f'stage1_expert0_gpu{gpu_id}', 'stage1_moe_aggregate')
        dot.edge(f'stage1_expert1_gpu{gpu_id}', 'stage1_moe_aggregate')
    
    dot.edge('stage1_moe_aggregate', 'stage1_moe_residual')
    dot.edge('stage1_moe_residual', 'comm_stage1_to_stage0')
    
    # Layer 2 connections
    dot.edge('comm_stage1_to_stage0', 'stage2_attn_qkv')
    dot.edge('stage2_attn_qkv', 'stage2_attn_score')
    dot.edge('stage2_attn_score', 'stage2_attn_out')
    dot.edge('stage2_attn_out', 'stage2_attn_residual')
    dot.edge('stage2_attn_residual', 'stage2_gate')
    
    for gpu_id in range(8):
        dot.edge('stage2_gate', f'stage2_expert0_gpu{gpu_id}', style='dashed', label='route')
        dot.edge('stage2_gate', f'stage2_expert1_gpu{gpu_id}', style='dashed', label='route')
        dot.edge(f'stage2_expert0_gpu{gpu_id}', 'stage2_moe_aggregate')
        dot.edge(f'stage2_expert1_gpu{gpu_id}', 'stage2_moe_aggregate')
    
    dot.edge('stage2_moe_aggregate', 'stage2_moe_residual')
    dot.edge('stage2_moe_residual', 'comm_stage2_to_stage1_again')
    
    # Layer 3 connections
    dot.edge('comm_stage2_to_stage1_again', 'stage3_attn_qkv')
    dot.edge('stage3_attn_qkv', 'stage3_attn_score')
    dot.edge('stage3_attn_score', 'stage3_attn_out')
    dot.edge('stage3_attn_out', 'stage3_attn_residual')
    dot.edge('stage3_attn_residual', 'stage3_gate')
    
    for gpu_id in range(8, 16):
        dot.edge('stage3_gate', f'stage3_expert0_gpu{gpu_id}', style='dashed', label='route')
        dot.edge('stage3_gate', f'stage3_expert1_gpu{gpu_id}', style='dashed', label='route')
        dot.edge(f'stage3_expert0_gpu{gpu_id}', 'stage3_moe_aggregate')
        dot.edge(f'stage3_expert1_gpu{gpu_id}', 'stage3_moe_aggregate')
    
    dot.edge('stage3_moe_aggregate', 'stage3_moe_residual')
    dot.edge('stage3_moe_residual', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/2025-09-05-09-22-08/baseline_moe_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-05-09-22-08/baseline_moe_dag.dot')
    print("Baseline DAG generated successfully")