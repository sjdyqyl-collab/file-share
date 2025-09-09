#!/usr/bin/env python3
"""
Generate DAG for the baseline deployment strategy
Tensor Parallelism (TP=8) + Pipeline Parallelism (PP=2) across 16 GPUs
"""

import graphviz

def create_baseline_dag():
    dot = graphviz.Digraph('baseline_tensor_pipeline_deployment', format='svg')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='1.0')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Model Input\nINPUT DIMENSION=[1024, seq_len, 8192]\nGPU=all', 
             shape='ellipse', fillcolor='lightgreen')
    
    # Stage 0: Layers 0-7 with tensor parallelism
    stage0_devices = list(range(8))
    prev_stage_output = 'input'
    
    for layer_id in range(8):
        # Create tensor parallel group for this layer
        tp_group = stage0_devices
        
        # Input split for tensor parallelism
        split_name = f'stage0_layer{layer_id}_split'
        dot.node(split_name, f'Split Tensor\nINPUT=[1024, seq_len, 8192]\nOUTPUT=[8, 1024, seq_len, 1024]\nGPUs={tp_group}', 
                 shape='parallelogram', fillcolor='lightyellow')
        dot.edge(prev_stage_output, split_name)
        
        # Layer norm (pre-attention) - replicated across TP group
        ln1_nodes = []
        for tp_rank, gpu_id in enumerate(tp_group):
            ln1_name = f'stage0_layer{layer_id}_ln1_tp{tp_rank}'
            dot.node(ln1_name, f'LayerNorm\nINPUT=[1024, seq_len, 1024]\nOUTPUT=[1024, seq_len, 1024]\nGPU={gpu_id}')
            dot.edge(split_name, ln1_name)
            ln1_nodes.append(ln1_name)
        
        # Multi-head attention with tensor parallelism
        attn_nodes = []
        for tp_rank, gpu_id in enumerate(tp_group):
            attn_name = f'stage0_layer{layer_id}_attn_tp{tp_rank}'
            dot.node(attn_name, f'MultiHeadAttention\nINPUT=[1024, seq_len, 1024]\nOUTPUT=[1024, seq_len, 1024]\nGPU={gpu_id}')
            dot.edge(ln1_nodes[tp_rank], attn_name)
            attn_nodes.append(attn_name)
        
        # All-reduce for attention output
        attn_reduce_name = f'stage0_layer{layer_id}_attn_reduce'
        dot.node(attn_reduce_name, f'All-Reduce Sum\nINPUT=[8, 1024, seq_len, 1024]\nOUTPUT=[1024, seq_len, 8192]\nGPUs={tp_group}', 
                 shape='parallelogram', fillcolor='lightyellow')
        for attn_node in attn_nodes:
            dot.edge(attn_node, attn_reduce_name)
        
        # Residual add after attention
        residual1_name = f'stage0_layer{layer_id}_residual1'
        dot.node(residual1_name, f'ResidualAdd\nINPUT1=[1024, seq_len, 8192]\nINPUT2=[1024, seq_len, 8192]\nOUTPUT=[1024, seq_len, 8192]\nGPUs={tp_group}')
        dot.edge(prev_stage_output, residual1_name)
        dot.edge(attn_reduce_name, residual1_name)
        
        # Split for FFN tensor parallelism
        ffn_split_name = f'stage0_layer{layer_id}_ffn_split'
        dot.node(ffn_split_name, f'Split Tensor\nINPUT=[1024, seq_len, 8192]\nOUTPUT=[8, 1024, seq_len, 1024]\nGPUs={tp_group}', 
                 shape='parallelogram', fillcolor='lightyellow')
        dot.edge(residual1_name, ffn_split_name)
        
        # Layer norm (pre-FFN) - replicated
        ln2_nodes = []
        for tp_rank, gpu_id in enumerate(tp_group):
            ln2_name = f'stage0_layer{layer_id}_ln2_tp{tp_rank}'
            dot.node(ln2_name, f'LayerNorm\nINPUT=[1024, seq_len, 1024]\nOUTPUT=[1024, seq_len, 1024]\nGPU={gpu_id}')
            dot.edge(ffn_split_name, ln2_name)
            ln2_nodes.append(ln2_name)
        
        # FFN with tensor parallelism
        ffn_nodes = []
        for tp_rank, gpu_id in enumerate(tp_group):
            ffn_name = f'stage0_layer{layer_id}_ffn_tp{tp_rank}'
            dot.node(ffn_name, f'FeedForward\nINPUT=[1024, seq_len, 1024]\nOUTPUT=[1024, seq_len, 1024]\nGPU={gpu_id}')
            dot.edge(ln2_nodes[tp_rank], ffn_name)
            ffn_nodes.append(ffn_name)
        
        # All-reduce for FFN output
        ffn_reduce_name = f'stage0_layer{layer_id}_ffn_reduce'
        dot.node(ffn_reduce_name, f'All-Reduce Sum\nINPUT=[8, 1024, seq_len, 1024]\nOUTPUT=[1024, seq_len, 8192]\nGPUs={tp_group}', 
                 shape='parallelogram', fillcolor='lightyellow')
        for ffn_node in ffn_nodes:
            dot.edge(ffn_node, ffn_reduce_name)
        
        # Residual add after FFN
        residual2_name = f'stage0_layer{layer_id}_residual2'
        dot.node(residual2_name, f'ResidualAdd\nINPUT1=[1024, seq_len, 8192]\nINPUT2=[1024, seq_len, 8192]\nOUTPUT=[1024, seq_len, 8192]\nGPUs={tp_group}')
        dot.edge(residual1_name, residual2_name)
        dot.edge(ffn_reduce_name, residual2_name)
        
        prev_stage_output = residual2_name
    
    # Pipeline communication between stage 0 and stage 1
    pipeline_comm = 'pipeline_stage0_to_stage1'
    dot.node(pipeline_comm, f'Pipeline Transfer\nINPUT=[1024, seq_len, 8192]\nOUTPUT=[1024, seq_len, 8192]\nFROM_GPUs=[0-7]\nTO_GPUs=[8-15]', 
             shape='parallelogram', fillcolor='orange')
    dot.edge(prev_stage_output, pipeline_comm)
    
    # Stage 1: Layers 8-15 with tensor parallelism
    stage1_devices = list(range(8, 16))
    prev_stage_output = pipeline_comm
    
    for layer_id in range(8, 16):
        actual_layer_id = layer_id  # 8-15
        tp_group = stage1_devices
        
        # Input split for tensor parallelism
        split_name = f'stage1_layer{actual_layer_id}_split'
        dot.node(split_name, f'Split Tensor\nINPUT=[1024, seq_len, 8192]\nOUTPUT=[8, 1024, seq_len, 1024]\nGPUs={tp_group}', 
                 shape='parallelogram', fillcolor='lightyellow')
        dot.edge(prev_stage_output, split_name)
        
        # Layer norm (pre-attention)
        ln1_nodes = []
        for tp_rank, gpu_id in enumerate(tp_group):
            ln1_name = f'stage1_layer{actual_layer_id}_ln1_tp{tp_rank}'
            dot.node(ln1_name, f'LayerNorm\nINPUT=[1024, seq_len, 1024]\nOUTPUT=[1024, seq_len, 1024]\nGPU={gpu_id}')
            dot.edge(split_name, ln1_name)
            ln1_nodes.append(ln1_name)
        
        # Multi-head attention with tensor parallelism
        attn_nodes = []
        for tp_rank, gpu_id in enumerate(tp_group):
            attn_name = f'stage1_layer{actual_layer_id}_attn_tp{tp_rank}'
            dot.node(attn_name, f'MultiHeadAttention\nINPUT=[1024, seq_len, 1024]\nOUTPUT=[1024, seq_len, 1024]\nGPU={gpu_id}')
            dot.edge(ln1_nodes[tp_rank], attn_name)
            attn_nodes.append(attn_name)
        
        # All-reduce for attention output
        attn_reduce_name = f'stage1_layer{actual_layer_id}_attn_reduce'
        dot.node(attn_reduce_name, f'All-Reduce Sum\nINPUT=[8, 1024, seq_len, 1024]\nOUTPUT=[1024, seq_len, 8192]\nGPUs={tp_group}', 
                 shape='parallelogram', fillcolor='lightyellow')
        for attn_node in attn_nodes:
            dot.edge(attn_node, attn_reduce_name)
        
        # Residual add after attention
        residual1_name = f'stage1_layer{actual_layer_id}_residual1'
        dot.node(residual1_name, f'ResidualAdd\nINPUT1=[1024, seq_len, 8192]\nINPUT2=[1024, seq_len, 8192]\nOUTPUT=[1024, seq_len, 8192]\nGPUs={tp_group}')
        dot.edge(prev_stage_output, residual1_name)
        dot.edge(attn_reduce_name, residual1_name)
        
        # Split for FFN tensor parallelism
        ffn_split_name = f'stage1_layer{actual_layer_id}_ffn_split'
        dot.node(ffn_split_name, f'Split Tensor\nINPUT=[1024, seq_len, 8192]\nOUTPUT=[8, 1024, seq_len, 1024]\nGPUs={tp_group}', 
                 shape='parallelogram', fillcolor='lightyellow')
        dot.edge(residual1_name, ffn_split_name)
        
        # Layer norm (pre-FFN)
        ln2_nodes = []
        for tp_rank, gpu_id in enumerate(tp_group):
            ln2_name = f'stage1_layer{actual_layer_id}_ln2_tp{tp_rank}'
            dot.node(ln2_name, f'LayerNorm\nINPUT=[1024, seq_len, 1024]\nOUTPUT=[1024, seq_len, 1024]\nGPU={gpu_id}')
            dot.edge(ffn_split_name, ln2_name)
            ln2_nodes.append(ln2_name)
        
        # FFN with tensor parallelism
        ffn_nodes = []
        for tp_rank, gpu_id in enumerate(tp_group):
            ffn_name = f'stage1_layer{actual_layer_id}_ffn_tp{tp_rank}'
            dot.node(ffn_name, f'FeedForward\nINPUT=[1024, seq_len, 1024]\nOUTPUT=[1024, seq_len, 1024]\nGPU={gpu_id}')
            dot.edge(ln2_nodes[tp_rank], ffn_name)
            ffn_nodes.append(ffn_name)
        
        # All-reduce for FFN output
        ffn_reduce_name = f'stage1_layer{actual_layer_id}_ffn_reduce'
        dot.node(ffn_reduce_name, f'All-Reduce Sum\nINPUT=[8, 1024, seq_len, 1024]\nOUTPUT=[1024, seq_len, 8192]\nGPUs={tp_group}', 
                 shape='parallelogram', fillcolor='lightyellow')
        for ffn_node in ffn_nodes:
            dot.edge(ffn_node, ffn_reduce_name)
        
        # Residual add after FFN
        residual2_name = f'stage1_layer{actual_layer_id}_residual2'
        dot.node(residual2_name, f'ResidualAdd\nINPUT1=[1024, seq_len, 8192]\nINPUT2=[1024, seq_len, 8192]\nOUTPUT=[1024, seq_len, 8192]\nGPUs={tp_group}')
        dot.edge(residual1_name, residual2_name)
        dot.edge(ffn_reduce_name, residual2_name)
        
        prev_stage_output = residual2_name
    
    # Output node
    dot.node('output', 'Model Output\nINPUT=[1024, seq_len, 8192]\nOUTPUT=[1024, seq_len, 8192]\nGPUs=[8-15]', 
             shape='ellipse', fillcolor='lightgreen')
    dot.edge(prev_stage_output, 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/2025-09-08-19-28-27/baseline_tensor_pipeline_deployment', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-08-19-28-27/baseline_tensor_pipeline_deployment.dot')
    print("Baseline tensor+pipeline deployment DAG generated successfully")