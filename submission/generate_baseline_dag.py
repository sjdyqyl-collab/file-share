#!/usr/bin/env python3
"""
Generate DAG for Baseline: Tensor Parallelism + Pipeline Parallelism
"""

import graphviz

def create_baseline_dag():
    dot = graphviz.Digraph('baseline_tensor_pipeline_parallelism', 
                          comment='Baseline: Tensor Parallelism + Pipeline Parallelism DAG',
                          node_attr={'shape': 'rectangle'})
    
    dot.attr(rankdir='TB', size='20,30', fontname='Arial')
    
    # Input layer
    dot.node('input', 'Input Tensor\nX: [1024, L, 8192]\nGPU: all',
             shape='parallelogram', style='filled', fillcolor='lightblue')
    
    # Pipeline Stage 1 (Layers 1-1)
    dot.node('stage1', 'Pipeline Stage 1\nLayers 1\nGPUs: 0-7',
             shape='ellipse', style='filled', fillcolor='lightyellow')
    
    # Pipeline Stage 2 (Layers 2)
    dot.node('stage2', 'Pipeline Stage 2\nLayers 2\nGPUs: 8-15',
             shape='ellipse', style='filled', fillcolor='lightyellow')
    
    # Tensor Parallelism within each stage
    for stage in [1, 2]:
        stage_gpus = list(range(8*(stage-1), 8*stage))
        
        # Input distribution for each stage
        dot.node(f'stage{stage}_input', 
                 f'Stage {stage} Input\n[1024, L, 8192] → [1024, L, 1024]\nGPUs: {stage_gpus}',
                 shape='parallelogram', style='filled', fillcolor='lightgreen')
        
        # Multi-Head Attention for each stage
        for tp_rank in range(8):
            gpu_id = 8*(stage-1) + tp_rank
            
            # Q projection (split across hidden dimension)
            dot.node(f'stage{stage}_q_{tp_rank}', 
                     f'Stage {stage} Q Projection\nW_Q: [1024, 1024]\nInput: [1024, L, 1024]\nOutput: [1024, L, 128]\nGPU: {gpu_id}',
                     shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # K projection
            dot.node(f'stage{stage}_k_{tp_rank}', 
                     f'Stage {stage} K Projection\nW_K: [1024, 1024]\nInput: [1024, L, 1024]\nOutput: [1024, L, 128]\nGPU: {gpu_id}',
                     shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # V projection
            dot.node(f'stage{stage}_v_{tp_rank}', 
                     f'Stage {stage} V Projection\nW_V: [1024, 1024]\nInput: [1024, L, 1024]\nOutput: [1024, L, 128]\nGPU: {gpu_id}',
                     shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # Attention computation
            dot.node(f'stage{stage}_attn_{tp_rank}', 
                     f'Stage {stage} Attention\nQ,K,V: [1024, L, 128]\nOutput: [1024, L, 128]\nGPU: {gpu_id}',
                     shape='rectangle', style='filled', fillcolor='lightpink')
            
            # FFN (since it's a dense transformer)
            dot.node(f'stage{stage}_ffn1_{tp_rank}', 
                     f'Stage {stage} FFN Layer1\nW1: [1024, 4096]\nInput: [1024, L, 1024]\nOutput: [1024, L, 4096]\nGPU: {gpu_id}',
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            dot.node(f'stage{stage}_ffn2_{tp_rank}', 
                     f'Stage {stage} FFN Layer2\nW2: [4096, 1024]\nInput: [1024, L, 4096]\nOutput: [1024, L, 1024]\nGPU: {gpu_id}',
                     shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # All-reduce operations
            dot.node(f'stage{stage}_allreduce_attn_{tp_rank}', 
                     f'Stage {stage} All-Reduce Attention\nInput: [1024, L, 128]\nOutput: [1024, L, 1024]\nGPU: {gpu_id}',
                     shape='ellipse', style='filled', fillcolor='orange')
            
            dot.node(f'stage{stage}_allreduce_ffn_{tp_rank}', 
                     f'Stage {stage} All-Reduce FFN\nInput: [1024, L, 1024]\nOutput: [1024, L, 1024]\nGPU: {gpu_id}',
                     shape='ellipse', style='filled', fillcolor='orange')
            
            # Residual connections
            dot.node(f'stage{stage}_residual1_{tp_rank}', 
                     f'Stage {stage} Residual Add\nInput1: [1024, L, 1024]\nInput2: [1024, L, 1024]\nOutput: [1024, L, 1024]\nGPU: {gpu_id}',
                     shape='diamond', style='filled', fillcolor='lightsteelblue')
            
            dot.node(f'stage{stage}_residual2_{tp_rank}', 
                     f'Stage {stage} Residual Add\nInput1: [1024, L, 1024]\nInput2: [1024, L, 1024]\nOutput: [1024, L, 1024]\nGPU: {gpu_id}',
                     shape='diamond', style='filled', fillcolor='lightsteelblue')
    
    # Pipeline communication
    dot.node('pipeline_comm', 
             'Pipeline Communication\nStage1 → Stage2\n[1024, L, 8192]\nGPU: 7→8',
             shape='ellipse', style='filled', fillcolor='gold')
    
    # Final output
    dot.node('output', 'Output Tensor\n[1024, L, 8192]\nGPU: all',
             shape='parallelogram', style='filled', fillcolor='lightblue')
    
    # Connections for Stage 1
    dot.edge('input', 'stage1')
    dot.edge('stage1', 'stage1_input')
    
    for tp_rank in range(8):
        gpu_id = tp_rank
        dot.edge('stage1_input', f'stage1_q_{tp_rank}')
        dot.edge('stage1_input', f'stage1_k_{tp_rank}')
        dot.edge('stage1_input', f'stage1_v_{tp_rank}')
        dot.edge(f'stage1_q_{tp_rank}', f'stage1_attn_{tp_rank}')
        dot.edge(f'stage1_k_{tp_rank}', f'stage1_attn_{tp_rank}')
        dot.edge(f'stage1_v_{tp_rank}', f'stage1_attn_{tp_rank}')
        dot.edge(f'stage1_attn_{tp_rank}', f'stage1_allreduce_attn_{tp_rank}')
        dot.edge(f'stage1_allreduce_attn_{tp_rank}', f'stage1_residual1_{tp_rank}')
        dot.edge('stage1_input', f'stage1_residual1_{tp_rank}')
        dot.edge(f'stage1_residual1_{tp_rank}', f'stage1_ffn1_{tp_rank}')
        dot.edge(f'stage1_ffn1_{tp_rank}', f'stage1_ffn2_{tp_rank}')
        dot.edge(f'stage1_ffn2_{tp_rank}', f'stage1_allreduce_ffn_{tp_rank}')
        dot.edge(f'stage1_allreduce_ffn_{tp_rank}', f'stage1_residual2_{tp_rank}')
        dot.edge(f'stage1_residual1_{tp_rank}', f'stage1_residual2_{tp_rank}')
    
    # Connections for Stage 2
    dot.edge('stage1_residual2_7', 'pipeline_comm')
    dot.edge('pipeline_comm', 'stage2')
    dot.edge('stage2', 'stage2_input')
    
    for tp_rank in range(8):
        gpu_id = 8 + tp_rank
        dot.edge('stage2_input', f'stage2_q_{tp_rank}')
        dot.edge('stage2_input', f'stage2_k_{tp_rank}')
        dot.edge('stage2_input', f'stage2_v_{tp_rank}')
        dot.edge(f'stage2_q_{tp_rank}', f'stage2_attn_{tp_rank}')
        dot.edge(f'stage2_k_{tp_rank}', f'stage2_attn_{tp_rank}')
        dot.edge(f'stage2_v_{tp_rank}', f'stage2_attn_{tp_rank}')
        dot.edge(f'stage2_attn_{tp_rank}', f'stage2_allreduce_attn_{tp_rank}')
        dot.edge(f'stage2_allreduce_attn_{tp_rank}', f'stage2_residual1_{tp_rank}')
        dot.edge('stage2_input', f'stage2_residual1_{tp_rank}')
        dot.edge(f'stage2_residual1_{tp_rank}', f'stage2_ffn1_{tp_rank}')
        dot.edge(f'stage2_ffn1_{tp_rank}', f'stage2_ffn2_{tp_rank}')
        dot.edge(f'stage2_ffn2_{tp_rank}', f'stage2_allreduce_ffn_{tp_rank}')
        dot.edge(f'stage2_allreduce_ffn_{tp_rank}', f'stage2_residual2_{tp_rank}')
        dot.edge(f'stage2_residual1_{tp_rank}', f'stage2_residual2_{tp_rank}')
    
    # Final output connection
    dot.edge('stage2_residual2_15', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/submission/baseline_tensor_pipeline_parallelism', format='svg', cleanup=True)
    dag.save('/home/wzc/data/file-share/submission/baseline_tensor_pipeline_parallelism.dot')