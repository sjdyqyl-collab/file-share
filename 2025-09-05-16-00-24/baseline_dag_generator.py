#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_baseline_dag():
    """Create baseline DAG with 16 GPUs, TP=8, PP=2, 4 experts/GPU"""
    dot = Digraph(comment='Baseline MoE Deployment - 16 GPUs')
    dot.attr(rankdir='TB', splines='ortho')
    
    # Global input
    dot.node('input', 'Input\n(batch=1024, seq=10K, hidden=32768)', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Pipeline stage 0 (layers 0-1) on GPUs 0-7
    with dot.subgraph(name='cluster_pipeline_0') as c0:
        c0.attr(label='Pipeline Stage 0 (Layers 0-1)\nGPUs 0-7', style='dashed')
        
        # Layer 0
        for gpu_id in range(8):
            gpu_cluster = f'cluster_gpu_{gpu_id}'
            with c0.subgraph(name=gpu_cluster) as gpu:
                gpu.attr(label=f'GPU {gpu_id} (4 experts)', style='solid')
                
                # MHA for layer 0
                mha_name = f'mha_0_gpu_{gpu_id}'
                gpu.node(mha_name, f'MHA Layer 0\nGPU {gpu_id}\nInput: (1024, 10K, 32768)\nOutput: (1024, 10K, 32768)', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # Expert selection gate for layer 0
                gate_name = f'gate_0_gpu_{gpu_id}'
                gpu.node(gate_name, f'Gate Layer 0\nGPU {gpu_id}\nSelect top-2 experts', 
                        shape='parallelogram', style='filled', fillcolor='yellow')
                
                # 4 experts on this GPU for layer 0
                for expert_id in range(4):
                    expert_name = f'expert_0_{expert_id}_gpu_{gpu_id}'
                    gpu.node(expert_name, f'Expert {expert_id}\nLayer 0\nGPU {gpu_id}\nInput: (tokens, 32768)\nOutput: (tokens, 32768)', 
                            shape='rectangle', style='filled', fillcolor='lightcoral')
                
                # FFN aggregation for layer 0
                ffn_name = f'ffn_0_gpu_{gpu_id}'
                gpu.node(ffn_name, f'FFN Layer 0\nGPU {gpu_id}\nAggregate experts', 
                        shape='parallelogram', style='filled', fillcolor='orange')
                
                # Residual connection for layer 0
                residual_name = f'residual_0_gpu_{gpu_id}'
                gpu.node(residual_name, f'Residual Add\nLayer 0\nGPU {gpu_id}\nInput: (1024, 10K, 32768)', 
                        shape='ellipse', style='filled', fillcolor='lightgray')
                
                # Layer 1 (same structure)
                mha_1_name = f'mha_1_gpu_{gpu_id}'
                gpu.node(mha_1_name, f'MHA Layer 1\nGPU {gpu_id}\nInput: (1024, 10K, 32768)\nOutput: (1024, 10K, 32768)', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                gate_1_name = f'gate_1_gpu_{gpu_id}'
                gpu.node(gate_1_name, f'Gate Layer 1\nGPU {gpu_id}\nSelect top-2 experts', 
                        shape='parallelogram', style='filled', fillcolor='yellow')
                
                for expert_id in range(4):
                    expert_1_name = f'expert_1_{expert_id}_gpu_{gpu_id}'
                    gpu.node(expert_1_name, f'Expert {expert_id}\nLayer 1\nGPU {gpu_id}\nInput: (tokens, 32768)\nOutput: (tokens, 32768)', 
                            shape='rectangle', style='filled', fillcolor='lightcoral')
                
                ffn_1_name = f'ffn_1_gpu_{gpu_id}'
                gpu.node(ffn_1_name, f'FFN Layer 1\nGPU {gpu_id}\nAggregate experts', 
                        shape='parallelogram', style='filled', fillcolor='orange')
                
                residual_1_name = f'residual_1_gpu_{gpu_id}'
                gpu.node(residual_1_name, f'Residual Add\nLayer 1\nGPU {gpu_id}\nInput: (1024, 10K, 32768)', 
                        shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Pipeline stage 1 (layers 2-3) on GPUs 8-15
    with dot.subgraph(name='cluster_pipeline_1') as c1:
        c1.attr(label='Pipeline Stage 1 (Layers 2-3)\nGPUs 8-15', style='dashed')
        
        # Layer 2
        for gpu_id in range(8, 16):
            gpu_cluster = f'cluster_gpu_{gpu_id}'
            with c1.subgraph(name=gpu_cluster) as gpu:
                gpu.attr(label=f'GPU {gpu_id} (4 experts)', style='solid')
                
                # MHA for layer 2
                mha_name = f'mha_2_gpu_{gpu_id}'
                gpu.node(mha_name, f'MHA Layer 2\nGPU {gpu_id}\nInput: (1024, 10K, 32768)\nOutput: (1024, 10K, 32768)', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # Expert selection gate for layer 2
                gate_name = f'gate_2_gpu_{gpu_id}'
                gpu.node(gate_name, f'Gate Layer 2\nGPU {gpu_id}\nSelect top-2 experts', 
                        shape='parallelogram', style='filled', fillcolor='yellow')
                
                # 4 experts on this GPU for layer 2
                for expert_id in range(4):
                    expert_name = f'expert_2_{expert_id}_gpu_{gpu_id}'
                    gpu.node(expert_name, f'Expert {expert_id}\nLayer 2\nGPU {gpu_id}\nInput: (tokens, 32768)\nOutput: (tokens, 32768)', 
                            shape='rectangle', style='filled', fillcolor='lightcoral')
                
                # FFN aggregation for layer 2
                ffn_name = f'ffn_2_gpu_{gpu_id}'
                gpu.node(ffn_name, f'FFN Layer 2\nGPU {gpu_id}\nAggregate experts', 
                        shape='parallelogram', style='filled', fillcolor='orange')
                
                # Residual connection for layer 2
                residual_name = f'residual_2_gpu_{gpu_id}'
                gpu.node(residual_name, f'Residual Add\nLayer 2\nGPU {gpu_id}\nInput: (1024, 10K, 32768)', 
                        shape='ellipse', style='filled', fillcolor='lightgray')
                
                # Layer 3 (same structure)
                mha_3_name = f'mha_3_gpu_{gpu_id}'
                gpu.node(mha_3_name, f'MHA Layer 3\nGPU {gpu_id}\nInput: (1024, 10K, 32768)\nOutput: (1024, 10K, 32768)', 
                        shape='rectangle', style='filled', fillcolor='lightgreen')
                
                gate_3_name = f'gate_3_gpu_{gpu_id}'
                gpu.node(gate_3_name, f'Gate Layer 3\nGPU {gpu_id}\nSelect top-2 experts', 
                        shape='parallelogram', style='filled', fillcolor='yellow')
                
                for expert_id in range(4):
                    expert_3_name = f'expert_3_{expert_id}_gpu_{gpu_id}'
                    gpu.node(expert_3_name, f'Expert {expert_id}\nLayer 3\nGPU {gpu_id}\nInput: (tokens, 32768)\nOutput: (tokens, 32768)', 
                            shape='rectangle', style='filled', fillcolor='lightcoral')
                
                ffn_3_name = f'ffn_3_gpu_{gpu_id}'
                gpu.node(ffn_3_name, f'FFN Layer 3\nGPU {gpu_id}\nAggregate experts', 
                        shape='parallelogram', style='filled', fillcolor='orange')
                
                residual_3_name = f'residual_3_gpu_{gpu_id}'
                gpu.node(residual_3_name, f'Residual Add\nLayer 3\nGPU {gpu_id}\nInput: (1024, 10K, 32768)', 
                        shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Global output
    dot.node('output', 'Output\n(batch=1024, seq=10K, hidden=32768)', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Add connections
    # Input to first layer
    for gpu_id in range(8):
        dot.edge('input', f'mha_0_gpu_{gpu_id}')
    
    # Layer 0 connections
    for gpu_id in range(8):
        dot.edge(f'mha_0_gpu_{gpu_id}', f'gate_0_gpu_{gpu_id}')
        dot.edge(f'gate_0_gpu_{gpu_id}', f'expert_0_0_gpu_{gpu_id}', style='dashed')
        dot.edge(f'gate_0_gpu_{gpu_id}', f'expert_0_1_gpu_{gpu_id}', style='dashed')
        dot.edge(f'gate_0_gpu_{gpu_id}', f'expert_0_2_gpu_{gpu_id}', style='dashed')
        dot.edge(f'gate_0_gpu_{gpu_id}', f'expert_0_3_gpu_{gpu_id}', style='dashed')
        
        for expert_id in range(4):
            dot.edge(f'expert_0_{expert_id}_gpu_{gpu_id}', f'ffn_0_gpu_{gpu_id}')
        
        dot.edge(f'ffn_0_gpu_{gpu_id}', f'residual_0_gpu_{gpu_id}')
        dot.edge(f'mha_0_gpu_{gpu_id}', f'residual_0_gpu_{gpu_id}')
        dot.edge(f'residual_0_gpu_{gpu_id}', f'mha_1_gpu_{gpu_id}')
    
    # Layer 1 connections
    for gpu_id in range(8):
        dot.edge(f'mha_1_gpu_{gpu_id}', f'gate_1_gpu_{gpu_id}')
        dot.edge(f'gate_1_gpu_{gpu_id}', f'expert_1_0_gpu_{gpu_id}', style='dashed')
        dot.edge(f'gate_1_gpu_{gpu_id}', f'expert_1_1_gpu_{gpu_id}', style='dashed')
        dot.edge(f'gate_1_gpu_{gpu_id}', f'expert_1_2_gpu_{gpu_id}', style='dashed')
        dot.edge(f'gate_1_gpu_{gpu_id}', f'expert_1_3_gpu_{gpu_id}', style='dashed')
        
        for expert_id in range(4):
            dot.edge(f'expert_1_{expert_id}_gpu_{gpu_id}', f'ffn_1_gpu_{gpu_id}')
        
        dot.edge(f'ffn_1_gpu_{gpu_id}', f'residual_1_gpu_{gpu_id}')
        dot.edge(f'mha_1_gpu_{gpu_id}', f'residual_1_gpu_{gpu_id}')
    
    # Pipeline communication between stage 0 and 1
    for gpu_id_src in range(8):
        for gpu_id_dst in range(8, 16):
            dot.edge(f'residual_1_gpu_{gpu_id_src}', f'mha_2_gpu_{gpu_id_dst}', 
                    label='Pipeline\nCommunication', style='dotted')
    
    # Layer 2 connections
    for gpu_id in range(8, 16):
        dot.edge(f'mha_2_gpu_{gpu_id}', f'gate_2_gpu_{gpu_id}')
        dot.edge(f'gate_2_gpu_{gpu_id}', f'expert_2_0_gpu_{gpu_id}', style='dashed')
        dot.edge(f'gate_2_gpu_{gpu_id}', f'expert_2_1_gpu_{gpu_id}', style='dashed')
        dot.edge(f'gate_2_gpu_{gpu_id}', f'expert_2_2_gpu_{gpu_id}', style='dashed')
        dot.edge(f'gate_2_gpu_{gpu_id}', f'expert_2_3_gpu_{gpu_id}', style='dashed')
        
        for expert_id in range(4):
            dot.edge(f'expert_2_{expert_id}_gpu_{gpu_id}', f'ffn_2_gpu_{gpu_id}')
        
        dot.edge(f'ffn_2_gpu_{gpu_id}', f'residual_2_gpu_{gpu_id}')
        dot.edge(f'mha_2_gpu_{gpu_id}', f'residual_2_gpu_{gpu_id}')
        dot.edge(f'residual_2_gpu_{gpu_id}', f'mha_3_gpu_{gpu_id}')
    
    # Layer 3 connections
    for gpu_id in range(8, 16):
        dot.edge(f'mha_3_gpu_{gpu_id}', f'gate_3_gpu_{gpu_id}')
        dot.edge(f'gate_3_gpu_{gpu_id}', f'expert_3_0_gpu_{gpu_id}', style='dashed')
        dot.edge(f'gate_3_gpu_{gpu_id}', f'expert_3_1_gpu_{gpu_id}', style='dashed')
        dot.edge(f'gate_3_gpu_{gpu_id}', f'expert_3_2_gpu_{gpu_id}', style='dashed')
        dot.edge(f'gate_3_gpu_{gpu_id}', f'expert_3_3_gpu_{gpu_id}', style='dashed')
        
        for expert_id in range(4):
            dot.edge(f'expert_3_{expert_id}_gpu_{gpu_id}', f'ffn_3_gpu_{gpu_id}')
        
        dot.edge(f'ffn_3_gpu_{gpu_id}', f'residual_3_gpu_{gpu_id}')
        dot.edge(f'mha_3_gpu_{gpu_id}', f'residual_3_gpu_{gpu_id}')
        dot.edge(f'residual_3_gpu_{gpu_id}', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/2025-09-05-16-00-24/baseline_moe_deployment', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-05-16-00-24/baseline_moe_deployment.dot')
    print("Baseline DAG generated successfully")