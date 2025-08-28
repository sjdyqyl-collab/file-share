#!/usr/bin/env python3
"""
Proposed DAG: Ring Attention + Sequence Parallelism
16 GPUs total: SP=16, each GPU handles 1/16th of sequence
"""

import graphviz

def create_proposed_dag():
    dot = graphviz.Digraph('proposed_dag', format='svg')
    dot.attr(rankdir='TB', size='25,35')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    dot.attr('edge', fontsize='10')
    
    # Input split across 16 GPUs
    dot.node('input_split', 'Input Split\n(B×L×d_model) → 16×(B×L/16×d_model)\nAll GPUs', shape='parallelogram', fillcolor='lightgreen')
    
    # Process for each GPU (showing GPU 0 as detailed example)
    for gpu_id in range(16):
        gpu_label = f'GPU {gpu_id}'
        
        # Create subgraph for each GPU
        with dot.subgraph(name=f'cluster_gpu{gpu_id}') as c:
            c.attr(label=f'{gpu_label} (Sequence Part {gpu_id}/16)', style='dashed')
            
            # Layer 0
            c.node(f'l0_qkv_proj_{gpu_id}', f'QKV Projection\n(B×L/16×d_model) → (B×L/16×3×d_model)\n{gpu_label}', fillcolor='yellow')
            c.node(f'l0_ring_attention_{gpu_id}', f'Ring Attention\n(B×L/16×d_model) → (B×L/16×d_model)\n{gpu_label}\n16 stages', fillcolor='lightcoral')
            c.node(f'l0_residual1_{gpu_id}', f'Add & Norm\n(B×L/16×d_model) + (B×L/16×d_model)\n{gpu_label}', fillcolor='lightgreen')
            c.node(f'l0_mlp_{gpu_id}', f'MLP\n(B×L/16×d_model) → (B×L/16×4×d_model) → (B×L/16×d_model)\n{gpu_label}', fillcolor='lightblue')
            c.node(f'l0_residual2_{gpu_id}', f'Add & Norm\n(B×L/16×d_model) + (B×L/16×d_model)\n{gpu_label}', fillcolor='lightgreen')
            
            # Layer 1
            c.node(f'l1_qkv_proj_{gpu_id}', f'QKV Projection\n(B×L/16×d_model) → (B×L/16×3×d_model)\n{gpu_label}', fillcolor='yellow')
            c.node(f'l1_ring_attention_{gpu_id}', f'Ring Attention\n(B×L/16×d_model) → (B×L/16×d_model)\n{gpu_label}\n16 stages', fillcolor='lightcoral')
            c.node(f'l1_residual1_{gpu_id}', f'Add & Norm\n(B×L/16×d_model) + (B×L/16×d_model)\n{gpu_label}', fillcolor='lightgreen')
            c.node(f'l1_mlp_{gpu_id}', f'MLP\n(B×L/16×d_model) → (B×L/16×4×d_model) → (B×L/16×d_model)\n{gpu_label}', fillcolor='lightblue')
            c.node(f'l1_residual2_{gpu_id}', f'Add & Norm\n(B×L/16×d_model) + (B×L/16×d_model)\n{gpu_label}', fillcolor='lightgreen')
            
            # Layer 2
            c.node(f'l2_qkv_proj_{gpu_id}', f'QKV Projection\n(B×L/16×d_model) → (B×L/16×3×d_model)\n{gpu_label}', fillcolor='yellow')
            c.node(f'l2_ring_attention_{gpu_id}', f'Ring Attention\n(B×L/16×d_model) → (B×L/16×d_model)\n{gpu_label}\n16 stages', fillcolor='lightcoral')
            c.node(f'l2_residual1_{gpu_id}', f'Add & Norm\n(B×L/16×d_model) + (B×L/16×d_model)\n{gpu_label}', fillcolor='lightgreen')
            c.node(f'l2_mlp_{gpu_id}', f'MLP\n(B×L/16×d_model) → (B×L/16×4×d_model) → (B×L/16×d_model)\n{gpu_label}', fillcolor='lightblue')
            c.node(f'l2_residual2_{gpu_id}', f'Add & Norm\n(B×L/16×d_model) + (B×L/16×d_model)\n{gpu_label}', fillcolor='lightgreen')
            
            # Layer 3
            c.node(f'l3_qkv_proj_{gpu_id}', f'QKV Projection\n(B×L/16×d_model) → (B×L/16×3×d_model)\n{gpu_label}', fillcolor='yellow')
            c.node(f'l3_ring_attention_{gpu_id}', f'Ring Attention\n(B×L/16×d_model) → (B×L/16×d_model)\n{gpu_label}\n16 stages', fillcolor='lightcoral')
            c.node(f'l3_residual1_{gpu_id}', f'Add & Norm\n(B×L/16×d_model) + (B×L/16×d_model)\n{gpu_label}', fillcolor='lightgreen')
            c.node(f'l3_mlp_{gpu_id}', f'MLP\n(B×L/16×d_model) → (B×L/16×4×d_model) → (B×L/16×d_model)\n{gpu_label}', fillcolor='lightblue')
            c.node(f'l3_residual2_{gpu_id}', f'Add & Norm\n(B×L/16×d_model) + (B×L/16×d_model)\n{gpu_label}', fillcolor='lightgreen')
    
    # Ring communication nodes for attention
    for stage in range(16):
        dot.node(f'ring_comm_{stage}', f'Ring Stage {stage}\nKV Block Transfer\n(B×L/16×d_model)\nGPU{(stage)%16}→GPU{(stage+1)%16}', shape='ellipse', fillcolor='orange')
    
    # Output aggregation
    dot.node('output_agg', 'Output Aggregation\n16×(B×L/16×d_model) → (B×L×d_model)\nAll GPUs → GPU0', shape='parallelogram', fillcolor='lightgreen')
    dot.node('output', 'Final Output\n(B×L×d_model)\nGPU0', shape='parallelogram', fillcolor='lightgreen')
    
    # Connections for GPU 0 (detailed)
    dot.edge('input_split', 'l0_qkv_proj_0')
    dot.edge('l0_qkv_proj_0', 'l0_ring_attention_0')
    
    # Ring attention connections for GPU 0
    for stage in range(16):
        dot.edge(f'l0_ring_attention_0', f'ring_comm_{stage}', style='dashed')
        dot.edge(f'ring_comm_{stage}', f'l0_ring_attention_0', style='dashed')
    
    dot.edge('l0_ring_attention_0', 'l0_residual1_0')
    dot.edge('l0_residual1_0', 'l0_mlp_0')
    dot.edge('l0_mlp_0', 'l0_residual2_0')
    
    dot.edge('l0_residual2_0', 'l1_qkv_proj_0')
    dot.edge('l1_qkv_proj_0', 'l1_ring_attention_0')
    
    # Ring attention for layer 1
    for stage in range(16):
        dot.edge(f'l1_ring_attention_0', f'ring_comm_{stage}', style='dashed')
        dot.edge(f'ring_comm_{stage}', f'l1_ring_attention_0', style='dashed')
    
    dot.edge('l1_ring_attention_0', 'l1_residual1_0')
    dot.edge('l1_residual1_0', 'l1_mlp_0')
    dot.edge('l1_mlp_0', 'l1_residual2_0')
    
    dot.edge('l1_residual2_0', 'l2_qkv_proj_0')
    dot.edge('l2_qkv_proj_0', 'l2_ring_attention_0')
    
    # Ring attention for layer 2
    for stage in range(16):
        dot.edge(f'l2_ring_attention_0', f'ring_comm_{stage}', style='dashed')
        dot.edge(f'ring_comm_{stage}', f'l2_ring_attention_0', style='dashed')
    
    dot.edge('l2_ring_attention_0', 'l2_residual1_0')
    dot.edge('l2_residual1_0', 'l2_mlp_0')
    dot.edge('l2_mlp_0', 'l2_residual2_0')
    
    dot.edge('l2_residual2_0', 'l3_qkv_proj_0')
    dot.edge('l3_qkv_proj_0', 'l3_ring_attention_0')
    
    # Ring attention for layer 3
    for stage in range(16):
        dot.edge(f'l3_ring_attention_0', f'ring_comm_{stage}', style='dashed')
        dot.edge(f'ring_comm_{stage}', f'l3_ring_attention_0', style='dashed')
    
    dot.edge('l3_ring_attention_0', 'l3_residual1_0')
    dot.edge('l3_residual1_0', 'l3_mlp_0')
    dot.edge('l3_mlp_0', 'l3_residual2_0')
    dot.edge('l3_residual2_0', 'output_agg')
    
    # Add connections for other GPUs (simplified representation)
    for gpu_id in range(1, 16):
        dot.edge('input_split', f'l0_qkv_proj_{gpu_id}')
        dot.edge(f'l0_qkv_proj_{gpu_id}', f'l0_ring_attention_{gpu_id}')
        dot.edge(f'l0_ring_attention_{gpu_id}', f'l0_residual1_{gpu_id}')
        dot.edge(f'l0_residual1_{gpu_id}', f'l0_mlp_{gpu_id}')
        dot.edge(f'l0_mlp_{gpu_id}', f'l0_residual2_{gpu_id}')
        
        dot.edge(f'l0_residual2_{gpu_id}', f'l1_qkv_proj_{gpu_id}')
        dot.edge(f'l1_qkv_proj_{gpu_id}', f'l1_ring_attention_{gpu_id}')
        dot.edge(f'l1_ring_attention_{gpu_id}', f'l1_residual1_{gpu_id}')
        dot.edge(f'l1_residual1_{gpu_id}', f'l1_mlp_{gpu_id}')
        dot.edge(f'l1_mlp_{gpu_id}', f'l1_residual2_{gpu_id}')
        
        dot.edge(f'l1_residual2_{gpu_id}', f'l2_qkv_proj_{gpu_id}')
        dot.edge(f'l2_qkv_proj_{gpu_id}', f'l2_ring_attention_{gpu_id}')
        dot.edge(f'l2_ring_attention_{gpu_id}', f'l2_residual1_{gpu_id}')
        dot.edge(f'l2_residual1_{gpu_id}', f'l2_mlp_{gpu_id}')
        dot.edge(f'l2_mlp_{gpu_id}', f'l2_residual2_{gpu_id}')
        
        dot.edge(f'l2_residual2_{gpu_id}', f'l3_qkv_proj_{gpu_id}')
        dot.edge(f'l3_qkv_proj_{gpu_id}', f'l3_ring_attention_{gpu_id}')
        dot.edge(f'l3_ring_attention_{gpu_id}', f'l3_residual1_{gpu_id}')
        dot.edge(f'l3_residual1_{gpu_id}', f'l3_mlp_{gpu_id}')
        dot.edge(f'l3_mlp_{gpu_id}', f'l3_residual2_{gpu_id}')
        dot.edge(f'l3_residual2_{gpu_id}', 'output_agg')
    
    dot.edge('output_agg', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_proposed_dag()
    dag.render('/home/wzc/data/file-share/submission/proposed_dag', cleanup=True)
    print("Proposed DAG saved to /home/wzc/data/file-share/submission/proposed_dag.svg")