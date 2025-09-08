#!/usr/bin/env python3

import graphviz

def create_baseline_dag():
    """
    Create baseline DAG with TP=8, PP=2, 16 GPUs total
    Each GPU has 4 experts + TP shards
    4 layers, 16 experts per layer = 64 experts total
    """
    dot = graphviz.Digraph('baseline_moe_deployment', 
                          comment='Baseline MoE Deployment: TP=8, PP=2, 16 GPUs')
    
    # Set graph attributes
    dot.attr(rankdir='TB', size='20,20')
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    
    # Input
    dot.node('input', 'Input\n[1024, 10000, 8192]', shape='ellipse', fillcolor='lightgreen')
    
    # Pipeline Stage 1 (Layers 0-1)
    with dot.subgraph(name='cluster_stage1') as stage1:
        stage1.attr(label='Pipeline Stage 1 (Layers 0-1)', style='dashed', color='red')
        
        # Layer 0
        with stage1.subgraph(name='cluster_layer0') as layer0:
            layer0.attr(label='Layer 0', style='dashed', color='blue')
            
            # MHA for layer 0 (split across 8 GPUs for TP=8)
            for gpu in range(8):
                layer0.node(f'mha0_tp{gpu}', f'MHA-0-TP{gpu}\n[1024, 10000, 8192/8]\nGPU{gpu}', 
                           fillcolor='lightcoral')
            
            # Expert 0-15 for layer 0 (4 experts per GPU, 4 GPUs)
            for expert_id in range(16):
                gpu_id = expert_id // 4
                layer0.node(f'expert0_{expert_id}', 
                           f'Expert-{expert_id}\n[1024, 10000, 32768]\nGPU{gpu_id}', 
                           fillcolor='lightyellow')
            
            # Gate for layer 0
            layer0.node('gate0', 'Gate-0\n[1024, 10000, 16]\nAll GPUs', 
                       shape='parallelogram', fillcolor='lightpink')
            
            # Residual add for layer 0
            layer0.node('residual0', 'Residual-0\n[1024, 10000, 8192]\nAll GPUs', 
                       fillcolor='lightgray')
        
        # Layer 1
        with stage1.subgraph(name='cluster_layer1') as layer1:
            layer1.attr(label='Layer 1', style='dashed', color='blue')
            
            # MHA for layer 1
            for gpu in range(8):
                layer1.node(f'mha1_tp{gpu}', f'MHA-1-TP{gpu}\n[1024, 10000, 8192/8]\nGPU{gpu}', 
                           fillcolor='lightcoral')
            
            # Expert 0-15 for layer 1 (4 experts per GPU, 4 GPUs)
            for expert_id in range(16):
                gpu_id = expert_id // 4
                layer1.node(f'expert1_{expert_id}', 
                           f'Expert-{expert_id}\n[1024, 10000, 32768]\nGPU{gpu_id}', 
                           fillcolor='lightyellow')
            
            # Gate for layer 1
            layer1.node('gate1', 'Gate-1\n[1024, 10000, 16]\nAll GPUs', 
                       shape='parallelogram', fillcolor='lightpink')
            
            # Residual add for layer 1
            layer1.node('residual1', 'Residual-1\n[1024, 10000, 8192]\nAll GPUs', 
                       fillcolor='lightgray')
    
    # Pipeline Stage 2 (Layers 2-3)
    with dot.subgraph(name='cluster_stage2') as stage2:
        stage2.attr(label='Pipeline Stage 2 (Layers 2-3)', style='dashed', color='red')
        
        # Layer 2
        with stage2.subgraph(name='cluster_layer2') as layer2:
            layer2.attr(label='Layer 2', style='dashed', color='blue')
            
            # MHA for layer 2
            for gpu in range(8, 16):
                layer2.node(f'mha2_tp{gpu-8}', f'MHA-2-TP{gpu-8}\n[1024, 10000, 8192/8]\nGPU{gpu}', 
                           fillcolor='lightcoral')
            
            # Expert 0-15 for layer 2 (4 experts per GPU, 4 GPUs)
            for expert_id in range(16):
                gpu_id = 8 + (expert_id // 4)
                layer2.node(f'expert2_{expert_id}', 
                           f'Expert-{expert_id}\n[1024, 10000, 32768]\nGPU{gpu_id}', 
                           fillcolor='lightyellow')
            
            # Gate for layer 2
            layer2.node('gate2', 'Gate-2\n[1024, 10000, 16]\nAll GPUs', 
                       shape='parallelogram', fillcolor='lightpink')
            
            # Residual add for layer 2
            layer2.node('residual2', 'Residual-2\n[1024, 10000, 8192]\nAll GPUs', 
                       fillcolor='lightgray')
        
        # Layer 3
        with stage2.subgraph(name='cluster_layer3') as layer3:
            layer3.attr(label='Layer 3', style='dashed', color='blue')
            
            # MHA for layer 3
            for gpu in range(8, 16):
                layer3.node(f'mha3_tp{gpu-8}', f'MHA-3-TP{gpu-8}\n[1024, 10000, 8192/8]\nGPU{gpu}', 
                           fillcolor='lightcoral')
            
            # Expert 0-15 for layer 3 (4 experts per GPU, 4 GPUs)
            for expert_id in range(16):
                gpu_id = 8 + (expert_id // 4)
                layer3.node(f'expert3_{expert_id}', 
                           f'Expert-{expert_id}\n[1024, 10000, 32768]\nGPU{gpu_id}', 
                           fillcolor='lightyellow')
            
            # Gate for layer 3
            layer3.node('gate3', 'Gate-3\n[1024, 10000, 16]\nAll GPUs', 
                       shape='parallelogram', fillcolor='lightpink')
            
            # Residual add for layer 3
            layer3.node('residual3', 'Residual-3\n[1024, 10000, 8192]\nAll GPUs', 
                       fillcolor='lightgray')
    
    # Output
    dot.node('output', 'Output\n[1024, 10000, 8192]', shape='ellipse', fillcolor='lightgreen')
    
    # Connections
    # Input to Layer 0 MHA
    for gpu in range(8):
        dot.edge('input', f'mha0_tp{gpu}')
    
    # Layer 0 connections
    for gpu in range(8):
        dot.edge(f'mha0_tp{gpu}', 'gate0')
    dot.edge('gate0', 'residual0', style='dashed')
    for expert_id in range(16):
        dot.edge('gate0', f'expert0_{expert_id}', style='dashed')
        dot.edge(f'expert0_{expert_id}', 'residual0')
    
    # Layer 0 to Layer 1
    dot.edge('residual0', 'gate1')
    for gpu in range(8):
        dot.edge('residual0', f'mha1_tp{gpu}')
    
    # Layer 1 connections
    for gpu in range(8):
        dot.edge(f'mha1_tp{gpu}', 'gate1')
    dot.edge('gate1', 'residual1', style='dashed')
    for expert_id in range(16):
        dot.edge('gate1', f'expert1_{expert_id}', style='dashed')
        dot.edge(f'expert1_{expert_id}', 'residual1')
    
    # Pipeline communication between stages
    dot.edge('residual1', 'gate2', label='Pipeline Send\nGPU0-7 to GPU8-15')
    
    # Layer 2 connections
    dot.edge('gate2', 'residual2', style='dashed')
    for gpu in range(8, 16):
        dot.edge('residual1', f'mha2_tp{gpu-8}')
    for gpu in range(8, 16):
        dot.edge(f'mha2_tp{gpu-8}', 'gate2')
    for expert_id in range(16):
        dot.edge('gate2', f'expert2_{expert_id}', style='dashed')
        dot.edge(f'expert2_{expert_id}', 'residual2')
    
    # Layer 2 to Layer 3
    dot.edge('residual2', 'gate3')
    for gpu in range(8, 16):
        dot.edge('residual2', f'mha3_tp{gpu-8}')
    
    # Layer 3 connections
    for gpu in range(8, 16):
        dot.edge(f'mha3_tp{gpu-8}', 'gate3')
    dot.edge('gate3', 'residual3', style='dashed')
    for expert_id in range(16):
        dot.edge('gate3', f'expert3_{expert_id}', style='dashed')
        dot.edge(f'expert3_{expert_id}', 'residual3')
    
    # Final output
    dot.edge('residual3', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_baseline_dag()
    dag.render('/home/wzc/data/file-share/2025-09-05-17-11-08/baseline_moe_deployment', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-05-17-11-08/baseline_moe_deployment.dot')
    print("Baseline DAG generated successfully")