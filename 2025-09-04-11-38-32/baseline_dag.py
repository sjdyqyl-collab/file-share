#!/usr/bin/env python3

import graphviz

def create_baseline_dag():
    """
    Create DAG for baseline: Tensor Parallel (TP=8) + Pipeline Parallel (PP=2)
    Total 16 GPUs organized as 2 pipeline stages with 8-way tensor parallelism
    """
    dot = graphviz.Digraph(comment='Dense Transformer Baseline DAG')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input\n(B=1024, L=16384, d=8192)\nAll GPUs', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Embedding layer - split across first 8 GPUs
    dot.node('embed_0_7', 'Embedding\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 0-7',
             shape='rectangle', fillcolor='lightyellow')
    
    # Pipeline Stage 0: Layers 0-1 on GPUs 0-7
    dot.attr('node', fillcolor='lightcoral')
    
    # Layer 0 - Multi-Head Attention
    dot.node('l0_attn_qkv_0', 'Layer0 MHA QKV\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 0-7',
             shape='rectangle')
    dot.node('l0_attn_score', 'Layer0 Attention Score\n(B=1024, H=16, L=16384, L=16384)\nTP=8\nGPUs 0-7',
             shape='rectangle')
    dot.node('l0_attn_out', 'Layer0 Attention Output\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 0-7',
             shape='rectangle')
    dot.node('l0_attn_res', 'Layer0 Residual Add\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 0-7',
             shape='diamond', fillcolor='lightgreen')
    
    # Layer 0 - MLP
    dot.node('l0_mlp_up', 'Layer0 MLP Up\n(B=1024, L=16384, d=32768)\nTP=8\nGPUs 0-7',
             shape='rectangle')
    dot.node('l0_mlp_down', 'Layer0 MLP Down\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 0-7',
             shape='rectangle')
    dot.node('l0_mlp_res', 'Layer0 MLP Residual Add\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 0-7',
             shape='diamond', fillcolor='lightgreen')
    
    # Layer 1 - Multi-Head Attention
    dot.node('l1_attn_qkv', 'Layer1 MHA QKV\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 0-7',
             shape='rectangle')
    dot.node('l1_attn_score', 'Layer1 Attention Score\n(B=1024, H=16, L=16384, L=16384)\nTP=8\nGPUs 0-7',
             shape='rectangle')
    dot.node('l1_attn_out', 'Layer1 Attention Output\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 0-7',
             shape='rectangle')
    dot.node('l1_attn_res', 'Layer1 Residual Add\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 0-7',
             shape='diamond', fillcolor='lightgreen')
    
    # Layer 1 - MLP
    dot.node('l1_mlp_up', 'Layer1 MLP Up\n(B=1024, L=16384, d=32768)\nTP=8\nGPUs 0-7',
             shape='rectangle')
    dot.node('l1_mlp_down', 'Layer1 MLP Down\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 0-7',
             shape='rectangle')
    dot.node('l1_mlp_res', 'Layer1 MLP Residual Add\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 0-7',
             shape='diamond', fillcolor='lightgreen')
    
    # Pipeline communication between stages
    dot.node('pipe_send', 'Pipeline Send\n(B=1024, L=16384, d=8192)\nGPUs 0-7 → GPUs 8-15',
             shape='ellipse', fillcolor='orange')
    dot.node('pipe_recv', 'Pipeline Receive\n(B=1024, L=16384, d=8192)\nGPUs 0-7 → GPUs 8-15',
             shape='ellipse', fillcolor='orange')
    
    # Pipeline Stage 1: Layers 2-3 on GPUs 8-15
    dot.attr('node', fillcolor='lightblue')
    
    # Layer 2 - Multi-Head Attention
    dot.node('l2_attn_qkv', 'Layer2 MHA QKV\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 8-15',
             shape='rectangle')
    dot.node('l2_attn_score', 'Layer2 Attention Score\n(B=1024, H=16, L=16384, L=16384)\nTP=8\nGPUs 8-15',
             shape='rectangle')
    dot.node('l2_attn_out', 'Layer2 Attention Output\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 8-15',
             shape='rectangle')
    dot.node('l2_attn_res', 'Layer2 Residual Add\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 8-15',
             shape='diamond', fillcolor='lightgreen')
    
    # Layer 2 - MLP
    dot.node('l2_mlp_up', 'Layer2 MLP Up\n(B=1024, L=16384, d=32768)\nTP=8\nGPUs 8-15',
             shape='rectangle')
    dot.node('l2_mlp_down', 'Layer2 MLP Down\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 8-15',
             shape='rectangle')
    dot.node('l2_mlp_res', 'Layer2 MLP Residual Add\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 8-15',
             shape='diamond', fillcolor='lightgreen')
    
    # Layer 3 - Multi-Head Attention
    dot.node('l3_attn_qkv', 'Layer3 MHA QKV\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 8-15',
             shape='rectangle')
    dot.node('l3_attn_score', 'Layer3 Attention Score\n(B=1024, H=16, L=16384, L=16384)\nTP=8\nGPUs 8-15',
             shape='rectangle')
    dot.node('l3_attn_out', 'Layer3 Attention Output\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 8-15',
             shape='rectangle')
    dot.node('l3_attn_res', 'Layer3 Residual Add\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 8-15',
             shape='diamond', fillcolor='lightgreen')
    
    # Layer 3 - MLP
    dot.node('l3_mlp_up', 'Layer3 MLP Up\n(B=1024, L=16384, d=32768)\nTP=8\nGPUs 8-15',
             shape='rectangle')
    dot.node('l3_mlp_down', 'Layer3 MLP Down\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 8-15',
             shape='rectangle')
    dot.node('l3_mlp_res', 'Layer3 MLP Residual Add\n(B=1024, L=16384, d=8192)\nTP=8\nGPUs 8-15',
             shape='diamond', fillcolor='lightgreen')
    
    # Output layer
    dot.node('lm_head', 'LM Head\n(B=1024, L=16384, V=32000)\nTP=8\nGPUs 8-15',
             shape='rectangle', fillcolor='lightyellow')
    dot.node('output', 'Output\n(B=1024, L=16384, V=32000)\nAll GPUs',
             shape='parallelogram', fillcolor='lightgreen')
    
    # Connect the nodes
    dot.edge('input', 'embed_0_7')
    
    # Layer 0
    dot.edge('embed_0_7', 'l0_attn_qkv_0')
    dot.edge('l0_attn_qkv_0', 'l0_attn_score')
    dot.edge('l0_attn_score', 'l0_attn_out')
    dot.edge('embed_0_7', 'l0_attn_res')
    dot.edge('l0_attn_out', 'l0_attn_res')
    dot.edge('l0_attn_res', 'l0_mlp_up')
    dot.edge('l0_mlp_up', 'l0_mlp_down')
    dot.edge('l0_attn_res', 'l0_mlp_res')
    dot.edge('l0_mlp_down', 'l0_mlp_res')
    
    # Layer 1
    dot.edge('l0_mlp_res', 'l1_attn_qkv')
    dot.edge('l1_attn_qkv', 'l1_attn_score')
    dot.edge('l1_attn_score', 'l1_attn_out')
    dot.edge('l0_mlp_res', 'l1_attn_res')
    dot.edge('l1_attn_out', 'l1_attn_res')
    dot.edge('l1_attn_res', 'l1_mlp_up')
    dot.edge('l1_mlp_up', 'l1_mlp_down')
    dot.edge('l1_attn_res', 'l1_mlp_res')
    dot.edge('l1_mlp_down', 'l1_mlp_res')
    
    # Pipeline communication
    dot.edge('l1_mlp_res', 'pipe_send')
    dot.edge('pipe_send', 'pipe_recv')
    dot.edge('pipe_recv', 'l2_attn_qkv')
    
    # Layer 2
    dot.edge('l2_attn_qkv', 'l2_attn_score')
    dot.edge('l2_attn_score', 'l2_attn_out')
    dot.edge('pipe_recv', 'l2_attn_res')
    dot.edge('l2_attn_out', 'l2_attn_res')
    dot.edge('l2_attn_res', 'l2_mlp_up')
    dot.edge('l2_mlp_up', 'l2_mlp_down')
    dot.edge('l2_attn_res', 'l2_mlp_res')
    dot.edge('l2_mlp_down', 'l2_mlp_res')
    
    # Layer 3
    dot.edge('l2_mlp_res', 'l3_attn_qkv')
    dot.edge('l3_attn_qkv', 'l3_attn_score')
    dot.edge('l3_attn_score', 'l3_attn_out')
    dot.edge('l2_mlp_res', 'l3_attn_res')
    dot.edge('l3_attn_out', 'l3_attn_res')
    dot.edge('l3_attn_res', 'l3_mlp_up')
    dot.edge('l3_mlp_up', 'l3_mlp_down')
    dot.edge('l3_attn_res', 'l3_mlp_res')
    dot.edge('l3_mlp_down', 'l3_mlp_res')
    
    # Output
    dot.edge('l3_mlp_res', 'lm_head')
    dot.edge('lm_head', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/2025-09-04-11-38-32/baseline_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-04-11-38-32/baseline_dag.dot')
    print("Baseline DAG generated successfully")