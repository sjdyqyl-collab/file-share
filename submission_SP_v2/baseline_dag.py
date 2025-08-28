#!/usr/bin/env python3
"""
Baseline DAG: Tensor Parallelism + Pipeline Parallelism
16 GPUs total: TP=8, PP=2
"""

import graphviz

def create_baseline_dag():
    dot = graphviz.Digraph('baseline_dag', format='svg')
    dot.attr(rankdir='TB', size='20,30')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    dot.attr('edge', fontsize='10')
    
    # Input
    dot.node('input', 'Input\n(B×L×d_model)\nAll GPUs', shape='parallelogram', fillcolor='lightgreen')
    
    # Pipeline Stage 0 (Layers 0-1) - GPUs 0-7
    with dot.subgraph(name='cluster_stage0') as c:
        c.attr(label='Pipeline Stage 0 (GPUs 0-7)', style='dashed')
        
        # Layer 0
        c.node('l0_qkv_proj', 'QKV Projection\n(B×L×d_model) → (B×L×3×d_model)\nTP=8 across GPUs 0-7', fillcolor='yellow')
        c.node('l0_attention', 'Multi-Head Attention\n(B×L×d_model) → (B×L×d_model)\nTP=8 across GPUs 0-7', fillcolor='lightcoral')
        c.node('l0_residual1', 'Add & Norm\n(B×L×d_model) + (B×L×d_model)\nTP=8 across GPUs 0-7', fillcolor='lightgreen')
        c.node('l0_mlp', 'MLP\n(B×L×d_model) → (B×L×4×d_model) → (B×L×d_model)\nTP=8 across GPUs 0-7', fillcolor='lightblue')
        c.node('l0_residual2', 'Add & Norm\n(B×L×d_model) + (B×L×d_model)\nTP=8 across GPUs 0-7', fillcolor='lightgreen')
        
        # Layer 1
        c.node('l1_qkv_proj', 'QKV Projection\n(B×L×d_model) → (B×L×3×d_model)\nTP=8 across GPUs 0-7', fillcolor='yellow')
        c.node('l1_attention', 'Multi-Head Attention\n(B×L×d_model) → (B×L×d_model)\nTP=8 across GPUs 0-7', fillcolor='lightcoral')
        c.node('l1_residual1', 'Add & Norm\n(B×L×d_model) + (B×L×d_model)\nTP=8 across GPUs 0-7', fillcolor='lightgreen')
        c.node('l1_mlp', 'MLP\n(B×L×d_model) → (B×L×4×d_model) → (B×L×d_model)\nTP=8 across GPUs 0-7', fillcolor='lightblue')
        c.node('l1_residual2', 'Add & Norm\n(B×L×d_model) + (B×L×d_model)\nTP=8 across GPUs 0-7', fillcolor='lightgreen')
    
    # Pipeline Stage 1 (Layers 2-3) - GPUs 8-15
    with dot.subgraph(name='cluster_stage1') as c:
        c.attr(label='Pipeline Stage 1 (GPUs 8-15)', style='dashed')
        
        # Layer 2
        c.node('l2_qkv_proj', 'QKV Projection\n(B×L×d_model) → (B×L×3×d_model)\nTP=8 across GPUs 8-15', fillcolor='yellow')
        c.node('l2_attention', 'Multi-Head Attention\n(B×L×d_model) → (B×L×d_model)\nTP=8 across GPUs 8-15', fillcolor='lightcoral')
        c.node('l2_residual1', 'Add & Norm\n(B×L×d_model) + (B×L×d_model)\nTP=8 across GPUs 8-15', fillcolor='lightgreen')
        c.node('l2_mlp', 'MLP\n(B×L×d_model) → (B×L×4×d_model) → (B×L×d_model)\nTP=8 across GPUs 8-15', fillcolor='lightblue')
        c.node('l2_residual2', 'Add & Norm\n(B×L×d_model) + (B×L×d_model)\nTP=8 across GPUs 8-15', fillcolor='lightgreen')
        
        # Layer 3
        c.node('l3_qkv_proj', 'QKV Projection\n(B×L×d_model) → (B×L×3×d_model)\nTP=8 across GPUs 8-15', fillcolor='yellow')
        c.node('l3_attention', 'Multi-Head Attention\n(B×L×d_model) → (B×L×d_model)\nTP=8 across GPUs 8-15', fillcolor='lightcoral')
        c.node('l3_residual1', 'Add & Norm\n(B×L×d_model) + (B×L×d_model)\nTP=8 across GPUs 8-15', fillcolor='lightgreen')
        c.node('l3_mlp', 'MLP\n(B×L×d_model) → (B×L×4×d_model) → (B×L×d_model)\nTP=8 across GPUs 8-15', fillcolor='lightblue')
        c.node('l3_residual2', 'Add & Norm\n(B×L×d_model) + (B×L×d_model)\nTP=8 across GPUs 8-15', fillcolor='lightgreen')
    
    # Communication nodes
    dot.node('tp_comm_0', 'Tensor Parallel\nAll-Reduce\n(B×L×d_model)\nGPUs 0-7', shape='ellipse', fillcolor='orange')
    dot.node('tp_comm_1', 'Tensor Parallel\nAll-Reduce\n(B×L×d_model)\nGPUs 8-15', shape='ellipse', fillcolor='orange')
    dot.node('pp_comm', 'Pipeline Parallel\nSend/Recv\n(B×L×d_model)\nGPU7↔GPU8', shape='ellipse', fillcolor='orange')
    
    # Output
    dot.node('output', 'Output\n(B×L×d_model)\nGPU15', shape='parallelogram', fillcolor='lightgreen')
    
    # Connections
    dot.edge('input', 'l0_qkv_proj')
    dot.edge('l0_qkv_proj', 'l0_attention')
    dot.edge('l0_attention', 'l0_residual1')
    dot.edge('l0_residual1', 'l0_mlp')
    dot.edge('l0_mlp', 'l0_residual2')
    dot.edge('l0_residual2', 'l1_qkv_proj')
    dot.edge('l1_qkv_proj', 'l1_attention')
    dot.edge('l1_attention', 'l1_residual1')
    dot.edge('l1_residual1', 'l1_mlp')
    dot.edge('l1_mlp', 'l1_residual2')
    dot.edge('l1_residual2', 'pp_comm')
    dot.edge('pp_comm', 'l2_qkv_proj')
    dot.edge('l2_qkv_proj', 'l2_attention')
    dot.edge('l2_attention', 'l2_residual1')
    dot.edge('l2_residual1', 'l2_mlp')
    dot.edge('l2_mlp', 'l2_residual2')
    dot.edge('l2_residual2', 'l3_qkv_proj')
    dot.edge('l3_qkv_proj', 'l3_attention')
    dot.edge('l3_attention', 'l3_residual1')
    dot.edge('l3_residual1', 'l3_mlp')
    dot.edge('l3_mlp', 'l3_residual2')
    dot.edge('l3_residual2', 'output')
    
    # Add tensor parallel communication connections
    dot.edge('l0_attention', 'tp_comm_0', style='dashed')
    dot.edge('tp_comm_0', 'l0_residual1', style='dashed')
    dot.edge('l1_attention', 'tp_comm_0', style='dashed')
    dot.edge('tp_comm_0', 'l1_residual1', style='dashed')
    dot.edge('l2_attention', 'tp_comm_1', style='dashed')
    dot.edge('tp_comm_1', 'l2_residual1', style='dashed')
    dot.edge('l3_attention', 'tp_comm_1', style='dashed')
    dot.edge('tp_comm_1', 'l3_residual1', style='dashed')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/submission/baseline_dag', cleanup=True)
    print("Baseline DAG saved to /home/wzc/data/file-share/submission/baseline_dag.svg")