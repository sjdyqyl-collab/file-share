#!/usr/bin/env python3

import graphviz

def create_baseline_dag():
    """
    Create DAG for baseline: Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2)
    Using 16 GPUs total: 8 GPUs per pipeline stage, 2 stages total
    """
    dot = graphviz.Digraph('baseline_tp8_pp2', comment='Baseline: TP8+PP2')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='1.0')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    dot.attr('edge', arrowhead='normal')
    
    # Model parameters from paper
    B = 1024  # batch size in tokens
    L = 1024  # sequence length (estimated from 1024 tokens/1024 batch = 1 token per sample)
    d_model = 8192
    H = 16  # attention heads
    d_h = d_model // H  # 512 per head
    ffn_hidden = 32768
    
    # Pipeline Stage 0: GPUs 0-7
    with dot.subgraph(name='cluster_stage0') as stage0:
        stage0.attr(label='Pipeline Stage 0 (GPUs 0-7)', style='rounded', fillcolor='lightgray')
        
        # Input for stage 0
        stage0.node('input0', f'Input\nX: {B}×{L//2}×{d_model}\n(on GPU 0-7)', shape='ellipse', fillcolor='lightgreen')
        
        # Layer 0
        stage0.node('layernorm0_0', f'LayerNorm\nIn: {B}×{L//2}×{d_model}\nOut: {B}×{L//2}×{d_model}\n(on all GPUs 0-7)', fillcolor='lightyellow')
        
        # MHA - Tensor Parallel across 8 GPUs
        # QKV projection (column parallel)
        stage0.node('qkv_proj_0', f'QKV Projection\nColumn Parallel\nW: {3*d_model}×{d_model}\nSplit: 3*{d_model//8}×{d_model} per GPU\n(on GPUs 0-7)', fillcolor='lightcoral')
        
        # Split heads for MHA
        stage0.node('split_heads_0', f'Split Heads\nIn: {B}×{L//2}×{3*d_model}\nOut: {B}×{H}×{L//2}×{d_h}\n(on GPUs 0-7)', fillcolor='lightcoral')
        
        # MHA computation (tensor parallel)
        stage0.node('mha_0', f'Multi-Head Attention\nTP across 8 GPUs\nEach GPU: {B}×{H//8}×{L//2}×{L//2}\n(on GPUs 0-7)', fillcolor='lightcoral')
        
        # Concat heads
        stage0.node('concat_heads_0', f'Concat Heads\nIn: {B}×{H}×{L//2}×{d_h}\nOut: {B}×{L//2}×{d_model}\n(on GPUs 0-7)', fillcolor='lightcoral')
        
        # Output projection (row parallel)
        stage0.node('out_proj_0', f'Output Projection\nRow Parallel\nW: {d_model}×{d_model}\nSplit: {d_model}×{d_model//8} per GPU\n(on GPUs 0-7)', fillcolor='lightcoral')
        
        # Residual connection
        stage0.node('residual_0', f'Residual Add\nIn: {B}×{L//2}×{d_model} (x2)\nOut: {B}×{L//2}×{d_model}\n(on GPUs 0-7)', fillcolor='lightblue')
        
        # MLP - Tensor Parallel across 8 GPUs
        stage0.node('layernorm1_0', f'LayerNorm\nIn: {B}×{L//2}×{d_model}\nOut: {B}×{L//2}×{d_model}\n(on all GPUs 0-7)', fillcolor='lightyellow')
        
        # MLP layer 1 (column parallel)
        stage0.node('mlp1_0', f'MLP Layer 1\nColumn Parallel\nW: {ffn_hidden}×{d_model}\nSplit: {ffn_hidden//8}×{d_model} per GPU\n(on GPUs 0-7)', fillcolor='lightcoral')
        
        # MLP activation
        stage0.node('gelu_0', f'GELU Activation\nIn: {B}×{L//2}×{ffn_hidden}\nOut: {B}×{L//2}×{ffn_hidden}\n(on GPUs 0-7)', fillcolor='lightyellow')
        
        # MLP layer 2 (row parallel)
        stage0.node('mlp2_0', f'MLP Layer 2\nRow Parallel\nW: {d_model}×{ffn_hidden}\nSplit: {d_model}×{ffn_hidden//8} per GPU\n(on GPUs 0-7)', fillcolor='lightcoral')
        
        # MLP residual
        stage0.node('residual_mlp_0', f'Residual Add\nIn: {B}×{L//2}×{d_model} (x2)\nOut: {B}×{L//2}×{d_model}\n(on GPUs 0-7)', fillcolor='lightblue')
        
        # Layer 1
        stage0.node('layernorm2_0', f'LayerNorm\nIn: {B}×{L//2}×{d_model}\nOut: {B}×{L//2}×{d_model}\n(on all GPUs 0-7)', fillcolor='lightyellow')
        
        stage0.node('qkv_proj_1_0', f'QKV Projection\nColumn Parallel\nW: {3*d_model}×{d_model}\n(on GPUs 0-7)', fillcolor='lightcoral')
        stage0.node('split_heads_1_0', f'Split Heads\n(on GPUs 0-7)', fillcolor='lightcoral')
        stage0.node('mha_1_0', f'Multi-Head Attention\nTP across 8 GPUs\n(on GPUs 0-7)', fillcolor='lightcoral')
        stage0.node('concat_heads_1_0', f'Concat Heads\n(on GPUs 0-7)', fillcolor='lightcoral')
        stage0.node('out_proj_1_0', f'Output Projection\nRow Parallel\n(on GPUs 0-7)', fillcolor='lightcoral')
        stage0.node('residual_1_0', f'Residual Add\n(on GPUs 0-7)', fillcolor='lightblue')
        
        stage0.node('layernorm3_0', f'LayerNorm\n(on all GPUs 0-7)', fillcolor='lightyellow')
        stage0.node('mlp1_1_0', f'MLP Layer 1\nColumn Parallel\n(on GPUs 0-7)', fillcolor='lightcoral')
        stage0.node('gelu_1_0', f'GELU Activation\n(on GPUs 0-7)', fillcolor='lightyellow')
        stage0.node('mlp2_1_0', f'MLP Layer 2\nRow Parallel\n(on GPUs 0-7)', fillcolor='lightcoral')
        stage0.node('residual_mlp_1_0', f'Residual Add\n(on GPUs 0-7)', fillcolor='lightblue')
    
    # Pipeline communication between stages
    dot.node('pipeline_comm', f'Pipeline Communication\nSend: {B}×{L//2}×{d_model}\n(from GPUs 0-7 to GPUs 8-15)', shape='parallelogram', fillcolor='orange')
    
    # Pipeline Stage 1: GPUs 8-15
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1 (GPUs 8-15)', style='rounded', fillcolor='lightgray')
        
        # Input for stage 1
        stage1.node('input1', f'Input\nX: {B}×{L//2}×{d_model}\n(on GPU 8-15)', shape='ellipse', fillcolor='lightgreen')
        
        # Layer 2
        stage1.node('layernorm0_1', f'LayerNorm\nIn: {B}×{L//2}×{d_model}\nOut: {B}×{L//2}×{d_model}\n(on all GPUs 8-15)', fillcolor='lightyellow')
        
        stage1.node('qkv_proj_2', f'QKV Projection\nColumn Parallel\nW: {3*d_model}×{d_model}\nSplit: 3*{d_model//8}×{d_model} per GPU\n(on GPUs 8-15)', fillcolor='lightcoral')
        stage1.node('split_heads_2', f'Split Heads\n(on GPUs 8-15)', fillcolor='lightcoral')
        stage1.node('mha_2', f'Multi-Head Attention\nTP across 8 GPUs\n(on GPUs 8-15)', fillcolor='lightcoral')
        stage1.node('concat_heads_2', f'Concat Heads\n(on GPUs 8-15)', fillcolor='lightcoral')
        stage1.node('out_proj_2', f'Output Projection\nRow Parallel\n(on GPUs 8-15)', fillcolor='lightcoral')
        stage1.node('residual_2', f'Residual Add\n(on GPUs 8-15)', fillcolor='lightblue')
        
        # MLP
        stage1.node('layernorm1_1', f'LayerNorm\n(on all GPUs 8-15)', fillcolor='lightyellow')
        stage1.node('mlp1_2', f'MLP Layer 1\nColumn Parallel\n(on GPUs 8-15)', fillcolor='lightcoral')
        stage1.node('gelu_2', f'GELU Activation\n(on GPUs 8-15)', fillcolor='lightyellow')
        stage1.node('mlp2_2', f'MLP Layer 2\nRow Parallel\n(on GPUs 8-15)', fillcolor='lightcoral')
        stage1.node('residual_mlp_2', f'Residual Add\n(on GPUs 8-15)', fillcolor='lightblue')
        
        # Layer 3
        stage1.node('layernorm2_1', f'LayerNorm\n(on all GPUs 8-15)', fillcolor='lightyellow')
        stage1.node('qkv_proj_3', f'QKV Projection\nColumn Parallel\n(on GPUs 8-15)', fillcolor='lightcoral')
        stage1.node('split_heads_3', f'Split Heads\n(on GPUs 8-15)', fillcolor='lightcoral')
        stage1.node('mha_3', f'Multi-Head Attention\nTP across 8 GPUs\n(on GPUs 8-15)', fillcolor='lightcoral')
        stage1.node('concat_heads_3', f'Concat Heads\n(on GPUs 8-15)', fillcolor='lightcoral')
        stage1.node('out_proj_3', f'Output Projection\nRow Parallel\n(on GPUs 8-15)', fillcolor='lightcoral')
        stage1.node('residual_3', f'Residual Add\n(on GPUs 8-15)', fillcolor='lightblue')
        
        stage1.node('layernorm3_1', f'LayerNorm\n(on all GPUs 8-15)', fillcolor='lightyellow')
        stage1.node('mlp1_3', f'MLP Layer 1\nColumn Parallel\n(on GPUs 8-15)', fillcolor='lightcoral')
        stage1.node('gelu_3', f'GELU Activation\n(on GPUs 8-15)', fillcolor='lightyellow')
        stage1.node('mlp2_3', f'MLP Layer 2\nRow Parallel\n(on GPUs 8-15)', fillcolor='lightcoral')
        stage1.node('residual_mlp_3', f'Residual Add\n(on GPUs 8-15)', fillcolor='lightblue')
        
        # Final output
        stage1.node('output', f'Output\nX: {B}×{L}×{d_model}\n(from GPUs 8-15)', shape='ellipse', fillcolor='lightgreen')
    
    # Connect the flow
    dot.edge('input0', 'layernorm0_0')
    dot.edge('layernorm0_0', 'qkv_proj_0')
    dot.edge('qkv_proj_0', 'split_heads_0')
    dot.edge('split_heads_0', 'mha_0')
    dot.edge('mha_0', 'concat_heads_0')
    dot.edge('concat_heads_0', 'out_proj_0')
    dot.edge('out_proj_0', 'residual_0')
    dot.edge('input0', 'residual_0')  # Residual connection
    
    dot.edge('residual_0', 'layernorm1_0')
    dot.edge('layernorm1_0', 'mlp1_0')
    dot.edge('mlp1_0', 'gelu_0')
    dot.edge('gelu_0', 'mlp2_0')
    dot.edge('mlp2_0', 'residual_mlp_0')
    dot.edge('residual_0', 'residual_mlp_0')  # Residual connection
    
    dot.edge('residual_mlp_0', 'layernorm2_0')
    dot.edge('layernorm2_0', 'qkv_proj_1_0')
    dot.edge('qkv_proj_1_0', 'split_heads_1_0')
    dot.edge('split_heads_1_0', 'mha_1_0')
    dot.edge('mha_1_0', 'concat_heads_1_0')
    dot.edge('concat_heads_1_0', 'out_proj_1_0')
    dot.edge('residual_mlp_0', 'residual_1_0')
    dot.edge('out_proj_1_0', 'residual_1_0')
    
    dot.edge('residual_1_0', 'layernorm3_0')
    dot.edge('layernorm3_0', 'mlp1_1_0')
    dot.edge('mlp1_1_0', 'gelu_1_0')
    dot.edge('gelu_1_0', 'mlp2_1_0')
    dot.edge('mlp2_1_0', 'residual_mlp_1_0')
    dot.edge('residual_1_0', 'residual_mlp_1_0')
    
    # Pipeline communication
    dot.edge('residual_mlp_1_0', 'pipeline_comm')
    dot.edge('pipeline_comm', 'input1')
    
    # Stage 1 connections
    dot.edge('input1', 'layernorm0_1')
    dot.edge('layernorm0_1', 'qkv_proj_2')
    dot.edge('qkv_proj_2', 'split_heads_2')
    dot.edge('split_heads_2', 'mha_2')
    dot.edge('mha_2', 'concat_heads_2')
    dot.edge('concat_heads_2', 'out_proj_2')
    dot.edge('input1', 'residual_2')
    dot.edge('out_proj_2', 'residual_2')
    
    dot.edge('residual_2', 'layernorm1_1')
    dot.edge('layernorm1_1', 'mlp1_2')
    dot.edge('mlp1_2', 'gelu_2')
    dot.edge('gelu_2', 'mlp2_2')
    dot.edge('mlp2_2', 'residual_mlp_2')
    dot.edge('residual_2', 'residual_mlp_2')
    
    dot.edge('residual_mlp_2', 'layernorm2_1')
    dot.edge('layernorm2_1', 'qkv_proj_3')
    dot.edge('qkv_proj_3', 'split_heads_3')
    dot.edge('split_heads_3', 'mha_3')
    dot.edge('mha_3', 'concat_heads_3')
    dot.edge('concat_heads_3', 'out_proj_3')
    dot.edge('residual_mlp_2', 'residual_3')
    dot.edge('out_proj_3', 'residual_3')
    
    dot.edge('residual_3', 'layernorm3_1')
    dot.edge('layernorm3_1', 'mlp1_3')
    dot.edge('mlp1_3', 'gelu_3')
    dot.edge('gelu_3', 'mlp2_3')
    dot.edge('mlp2_3', 'residual_mlp_3')
    dot.edge('residual_3', 'residual_mlp_3')
    dot.edge('residual_mlp_3', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/2025-09-04-10-18-25/baseline_tp8_pp2', format='svg', cleanup=True)
    dag.save('/home/wzc/data/file-share/2025-09-04-10-18-25/baseline_tp8_pp2.dot')
    print("Baseline DAG generated successfully")