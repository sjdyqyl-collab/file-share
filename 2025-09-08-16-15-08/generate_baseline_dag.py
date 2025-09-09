#!/usr/bin/env python3

import graphviz

def create_baseline_dag():
    """Create baseline DAG with 16 GPUs, TP=8, PP=2, 4 experts/GPU"""
    dot = graphviz.Digraph('baseline_moe_dag', format='svg')
    dot.attr(rankdir='TB', size='30,20')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input\\n[Batch: 1024×10000, Dim: 8192]', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Pipeline Stage 0 (GPUs 0-7)
    with dot.subgraph(name='cluster_pipeline_stage_0') as c0:
        c0.attr(label='Pipeline Stage 0 (GPUs 0-7)', style='dashed', color='blue')
        
        # Layer 0 on Stage 0
        with c0.subgraph(name='cluster_layer_0') as layer0:
            layer0.attr(label='MoE Layer 0', style='rounded')
            
            # Gating network (distributed across TP=8)
            layer0.node('gate_0', 'Gating Network Layer 0\\n[Input: 8192, Output: 16 experts]', 
                       shape='diamond', fillcolor='yellow')
            
            # Experts 0-3 on GPUs 0-3 (4 experts/GPU)
            for gpu_id in range(4):
                gpu_cluster = f'cluster_gpu_{gpu_id}'
                with layer0.subgraph(name=gpu_cluster) as gpu:
                    gpu.attr(label=f'GPU {gpu_id} (4 experts)', style='dotted')
                    
                    for expert_idx in range(4):
                        expert_id = gpu_id * 4 + expert_idx
                        gpu.node(f'expert_0_{expert_id}', 
                                f'Expert {expert_id}\\n[MLP: 8192→32768→8192]\\nGPU {gpu_id}',
                                shape='rectangle', fillcolor='lightcoral')
            
            # Tensor parallel operations for each expert
            for expert_id in range(16):
                gpu_id = expert_id // 4
                # TP operations for expert
                for tp_rank in range(8):
                    dot.node(f'tp_0_{expert_id}_{tp_rank}', 
                            f'TP Rank {tp_rank}\\nExpert {expert_id}\\nGPU {gpu_id}',
                            shape='ellipse', fillcolor='lightblue', width='1.2')
    
    # Pipeline Stage 1 (GPUs 8-15)
    with dot.subgraph(name='cluster_pipeline_stage_1') as c1:
        c1.attr(label='Pipeline Stage 1 (GPUs 8-15)', style='dashed', color='red')
        
        # Layer 0 on Stage 1
        with c1.subgraph(name='cluster_layer_0_stage1') as layer0_s1:
            layer0_s1.attr(label='MoE Layer 0 Continued', style='rounded')
            
            # Experts 16-31 on GPUs 4-7 (but mapped to GPUs 8-11 in stage 1)
            for gpu_id in range(4):
                actual_gpu = gpu_id + 8
                gpu_cluster = f'cluster_gpu_{actual_gpu}'
                with layer0_s1.subgraph(name=gpu_cluster) as gpu:
                    gpu.attr(label=f'GPU {actual_gpu} (4 experts)', style='dotted')
                    
                    for expert_idx in range(4):
                        expert_id = 16 + gpu_id * 4 + expert_idx
                        gpu.node(f'expert_0_{expert_id}', 
                                f'Expert {expert_id}\\n[MLP: 8192→32768→8192]\\nGPU {actual_gpu}',
                                shape='rectangle', fillcolor='lightcoral')
    
    # Add remaining layers (1-3) with similar structure
    for layer_id in range(1, 4):
        stage = layer_id % 2
        gpu_offset = 0 if stage == 0 else 8
        
        with dot.subgraph(name=f'cluster_layer_{layer_id}') as layer:
            layer.attr(label=f'MoE Layer {layer_id}', style='rounded')
            
            # Gating network
            layer.node(f'gate_{layer_id}', f'Gating Network Layer {layer_id}\\n[Input: 8192, Output: 16 experts]', 
                      shape='diamond', fillcolor='yellow')
            
            # Experts for this layer
            for gpu_id in range(8):
                actual_gpu = gpu_id + gpu_offset
                for expert_idx in range(4):
                    expert_num = layer_id * 16 + gpu_id * 4 + expert_idx
                    layer.node(f'expert_{layer_id}_{expert_num}', 
                              f'Expert {expert_num}\\n[MLP: 8192→32768→8192]\\nGPU {actual_gpu}',
                              shape='rectangle', fillcolor='lightcoral')
    
    # Add communication nodes
    dot.node('tp_allreduce', 'TP All-Reduce\\n[Across 8 GPUs]', 
             shape='parallelogram', fillcolor='orange')
    dot.node('pp_send', 'Pipeline Send\\n[Stage 0→1]', 
             shape='parallelogram', fillcolor='purple')
    dot.node('pp_recv', 'Pipeline Receive\\n[Stage 1←0]', 
             shape='parallelogram', fillcolor='purple')
    
    # Output node
    dot.node('output', 'Output\\n[Batch: 1024×10000, Dim: 8192]', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Add edges
    # Input to first gating
    dot.edge('input', 'gate_0')
    
    # Gating to experts with routing
    for expert_id in range(16):
        dot.edge('gate_0', f'expert_0_{expert_id}', style='dashed', label='route')
    
    # Expert computation with TP
    for expert_id in range(16):
        gpu_id = expert_id // 4
        for tp_rank in range(8):
            dot.edge(f'expert_0_{expert_id}', f'tp_0_{expert_id}_{tp_rank}')
    
    # TP all-reduce
    for expert_id in range(16):
        for tp_rank in range(8):
            dot.edge(f'tp_0_{expert_id}_{tp_rank}', 'tp_allreduce')
    
    # Pipeline communication
    dot.edge('tp_allreduce', 'pp_send')
    dot.edge('pp_send', 'pp_recv')
    dot.edge('pp_recv', 'gate_1')
    
    # Continue for all layers
    for layer_id in range(1, 4):
        dot.edge(f'gate_{layer_id}', f'expert_{layer_id}_{layer_id*16}')
        # Simplified: connect to first expert of each layer
        if layer_id < 3:
            dot.edge(f'expert_{layer_id}_{layer_id*16}', f'gate_{layer_id+1}')
    
    # Final output
    dot.edge('expert_3_63', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/2025-09-08-16-15-08/baseline_moe_dag', format='svg', cleanup=False)
    
    # Save DOT file
    with open('/home/wzc/data/file-share/2025-09-08-16-15-08/baseline_moe_dag.dot', 'w') as f:
        f.write(dag.source)
    
    print("Baseline DAG generated successfully!")
    print(f"SVG: /home/wzc/data/file-share/2025-09-08-16-15-08/baseline_moe_dag.svg")
    print(f"DOT: /home/wzc/data/file-share/2025-09-08-16-15-08/baseline_moe_dag.dot")