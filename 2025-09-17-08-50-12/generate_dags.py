#!/usr/bin/env python3

import os
from graphviz import Digraph

# Create the output directory if it doesn't exist
output_dir = "/home/wzc/data/file-share/2025-09-17-08-50-12"
os.makedirs(output_dir, exist_ok=True)

def create_baseline_dag():
    """Create baseline DAG with TP=8, PP=2"""
    dot = Digraph(comment='Baseline MoE Deployment (TP=8, PP=2)')
    dot.attr(rankdir='TB', size='20,20')
    
    # Input
    dot.node('input', 'Input\n[batch_size=1024, seq_len=10000, hidden=8192]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Pipeline Stage 0 (Layers 0-1)
    with dot.subgraph(name='cluster_stage0') as c:
        c.attr(label='Pipeline Stage 0\n(GPUs 0-7)', style='dashed')
        
        # Layer 0
        c.node('layer0_mha', 'Multi-Head Attention\n[batch=1024, seq=10000, heads=16, d_k=512]\nTP=8\nGPU: 0-7', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c.node('layer0_residual1', 'Residual Add\n[batch=1024, seq=10000, hidden=8192]\nGPU: 0-7', 
               shape='parallelogram', style='filled', fillcolor='lightgray')
        c.node('layer0_mlp', 'MoE Layer 0\n8 experts per GPU\n[batch=1024, seq=10000, hidden=8192->32768->8192]\nTP=8\nGPU: 0-7', 
               shape='rectangle', style='filled', fillcolor='lightyellow')
        c.node('layer0_residual2', 'Residual Add\n[batch=1024, seq=10000, hidden=8192]\nGPU: 0-7', 
               shape='parallelogram', style='filled', fillcolor='lightgray')
        
        # Layer 1
        c.node('layer1_mha', 'Multi-Head Attention\n[batch=1024, seq=10000, heads=16, d_k=512]\nTP=8\nGPU: 0-7', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c.node('layer1_residual1', 'Residual Add\n[batch=1024, seq=10000, hidden=8192]\nGPU: 0-7', 
               shape='parallelogram', style='filled', fillcolor='lightgray')
        c.node('layer1_mlp', 'MoE Layer 1\n8 experts per GPU\n[batch=1024, seq=10000, hidden=8192->32768->8192]\nTP=8\nGPU: 0-7', 
               shape='rectangle', style='filled', fillcolor='lightyellow')
        c.node('layer1_residual2', 'Residual Add\n[batch=1024, seq=10000, hidden=8192]\nGPU: 0-7', 
               shape='parallelogram', style='filled', fillcolor='lightgray')
    
    # Pipeline communication
    dot.node('pipe_comm1', 'Pipeline Communication\n[batch=1024, seq=10000, hidden=8192]\nGPU: 7 -> 8', 
             shape='ellipse', style='filled', fillcolor='lightcoral')
    
    # Pipeline Stage 1 (Layers 2-3)
    with dot.subgraph(name='cluster_stage1') as c:
        c.attr(label='Pipeline Stage 1\n(GPUs 8-15)', style='dashed')
        
        # Layer 2
        c.node('layer2_mha', 'Multi-Head Attention\n[batch=1024, seq=10000, heads=16, d_k=512]\nTP=8\nGPU: 8-15', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c.node('layer2_residual1', 'Residual Add\n[batch=1024, seq=10000, hidden=8192]\nGPU: 8-15', 
               shape='parallelogram', style='filled', fillcolor='lightgray')
        c.node('layer2_mlp', 'MoE Layer 2\n8 experts per GPU\n[batch=1024, seq=10000, hidden=8192->32768->8192]\nTP=8\nGPU: 8-15', 
               shape='rectangle', style='filled', fillcolor='lightyellow')
        c.node('layer2_residual2', 'Residual Add\n[batch=1024, seq=10000, hidden=8192]\nGPU: 8-15', 
               shape='parallelogram', style='filled', fillcolor='lightgray')
        
        # Layer 3
        c.node('layer3_mha', 'Multi-Head Attention\n[batch=1024, seq=10000, heads=16, d_k=512]\nTP=8\nGPU: 8-15', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c.node('layer3_residual1', 'Residual Add\n[batch=1024, seq=10000, hidden=8192]\nGPU: 8-15', 
               shape='parallelogram', style='filled', fillcolor='lightgray')
        c.node('layer3_mlp', 'MoE Layer 3\n8 experts per GPU\n[batch=1024, seq=10000, hidden=8192->32768->8192]\nTP=8\nGPU: 8-15', 
               shape='rectangle', style='filled', fillcolor='lightyellow')
        c.node('layer3_residual2', 'Residual Add\n[batch=1024, seq=10000, hidden=8192]\nGPU: 8-15', 
               shape='parallelogram', style='filled', fillcolor='lightgray')
    
    # Output
    dot.node('output', 'Output\n[batch_size=1024, seq_len=10000, hidden=8192]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connections
    dot.edge('input', 'layer0_mha')
    dot.edge('layer0_mha', 'layer0_residual1')
    dot.edge('layer0_residual1', 'layer0_mlp')
    dot.edge('layer0_mlp', 'layer0_residual2')
    dot.edge('layer0_residual2', 'layer1_mha')
    dot.edge('layer1_mha', 'layer1_residual1')
    dot.edge('layer1_residual1', 'layer1_mlp')
    dot.edge('layer1_residual2', 'pipe_comm1')
    dot.edge('pipe_comm1', 'layer2_mha')
    dot.edge('layer2_mha', 'layer2_residual1')
    dot.edge('layer2_residual1', 'layer2_mlp')
    dot.edge('layer2_mlp', 'layer2_residual2')
    dot.edge('layer2_residual2', 'layer3_mha')
    dot.edge('layer3_mha', 'layer3_residual1')
    dot.edge('layer3_residual1', 'layer3_mlp')
    dot.edge('layer3_mlp', 'layer3_residual2')
    dot.edge('layer3_residual2', 'output')
    
    # Add residual connections
    dot.edge('input', 'layer0_residual1', style='dashed')
    dot.edge('layer0_residual1', 'layer0_residual2', style='dashed')
    dot.edge('layer0_residual2', 'layer1_residual1', style='dashed')
    dot.edge('layer1_residual1', 'layer1_residual2', style='dashed')
    dot.edge('pipe_comm1', 'layer2_residual1', style='dashed')
    dot.edge('layer2_residual1', 'layer2_residual2', style='dashed')
    dot.edge('layer2_residual2', 'layer3_residual1', style='dashed')
    dot.edge('layer3_residual1', 'layer3_residual2', style='dashed')
    
    return dot

def create_proposed_dag():
    """Create proposed DAG with cross-node expert parallelism"""
    dot = Digraph(comment='Proposed Cross-Node Expert Parallelism')
    dot.attr(rankdir='TB', size='20,20')
    
    # Input
    dot.node('input', 'Input\n[batch_size=1024, seq_len=10000, hidden=8192]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Gating network for each layer
    dot.node('gate0', 'Gating Network Layer 0\n[batch=1024, seq=10000, experts=16]\nGPU: 0-15', 
             shape='parallelogram', style='filled', fillcolor='lightcoral')
    dot.node('gate1', 'Gating Network Layer 1\n[batch=1024, seq=10000, experts=16]\nGPU: 0-15', 
             shape='parallelogram', style='filled', fillcolor='lightcoral')
    dot.node('gate2', 'Gating Network Layer 2\n[batch=1024, seq=10000, experts=16]\nGPU: 0-15', 
             shape='parallelogram', style='filled', fillcolor='lightcoral')
    dot.node('gate3', 'Gating Network Layer 3\n[batch=1024, seq=10000, experts=16]\nGPU: 0-15', 
             shape='parallelogram', style='filled', fillcolor='lightcoral')
    
    # Layer 0 - Expert Parallelism
    with dot.subgraph(name='cluster_layer0') as c:
        c.attr(label='Layer 0 - Expert Parallelism', style='dashed')
        
        # MHA for layer 0
        c.node('layer0_mha', 'Multi-Head Attention\n[batch=1024, seq=10000, heads=16, d_k=512]\nGPU: 0-15', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c.node('layer0_residual1', 'Residual Add\n[batch=1024, seq=10000, hidden=8192]\nGPU: 0-15', 
               shape='parallelogram', style='filled', fillcolor='lightgray')
        
        # 16 experts for layer 0 - one per GPU
        for i in range(16):
            c.node(f'layer0_expert{i}', f'Expert {i}\n[batch=variable, seq=variable, hidden=8192->32768->8192]\nGPU: {i}', 
                   shape='rectangle', style='filled', fillcolor='lightyellow')
        
        c.node('layer0_aggregate', 'Aggregate Expert Outputs\n[batch=1024, seq=10000, hidden=8192]\nGPU: 0-15', 
               shape='parallelogram', style='filled', fillcolor='lightcoral')
        c.node('layer0_residual2', 'Residual Add\n[batch=1024, seq=10000, hidden=8192]\nGPU: 0-15', 
               shape='parallelogram', style='filled', fillcolor='lightgray')
    
    # Layer 1 - Expert Parallelism
    with dot.subgraph(name='cluster_layer1') as c:
        c.attr(label='Layer 1 - Expert Parallelism', style='dashed')
        
        # MHA for layer 1
        c.node('layer1_mha', 'Multi-Head Attention\n[batch=1024, seq=10000, heads=16, d_k=512]\nGPU: 0-15', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c.node('layer1_residual1', 'Residual Add\n[batch=1024, seq=10000, hidden=8192]\nGPU: 0-15', 
               shape='parallelogram', style='filled', fillcolor='lightgray')
        
        # 16 experts for layer 1 - one per GPU
        for i in range(16):
            c.node(f'layer1_expert{i}', f'Expert {i}\n[batch=variable, seq=variable, hidden=8192->32768->8192]\nGPU: {i}', 
                   shape='rectangle', style='filled', fillcolor='lightyellow')
        
        c.node('layer1_aggregate', 'Aggregate Expert Outputs\n[batch=1024, seq=10000, hidden=8192]\nGPU: 0-15', 
               shape='parallelogram', style='filled', fillcolor='lightcoral')
        c.node('layer1_residual2', 'Residual Add\n[batch=1024, seq=10000, hidden=8192]\nGPU: 0-15', 
               shape='parallelogram', style='filled', fillcolor='lightgray')
    
    # Layer 2 - Expert Parallelism
    with dot.subgraph(name='cluster_layer2') as c:
        c.attr(label='Layer 2 - Expert Parallelism', style='dashed')
        
        # MHA for layer 2
        c.node('layer2_mha', 'Multi-Head Attention\n[batch=1024, seq=10000, heads=16, d_k=512]\nGPU: 0-15', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c.node('layer2_residual1', 'Residual Add\n[batch=1024, seq=10000, hidden=8192]\nGPU: 0-15', 
               shape='parallelogram', style='filled', fillcolor='lightgray')
        
        # 16 experts for layer 2 - one per GPU
        for i in range(16):
            c.node(f'layer2_expert{i}', f'Expert {i}\n[batch=variable, seq=variable, hidden=8192->32768->8192]\nGPU: {i}', 
                   shape='rectangle', style='filled', fillcolor='lightyellow')
        
        c.node('layer2_aggregate', 'Aggregate Expert Outputs\n[batch=1024, seq=10000, hidden=8192]\nGPU: 0-15', 
               shape='parallelogram', style='filled', fillcolor='lightcoral')
        c.node('layer2_residual2', 'Residual Add\n[batch=1024, seq=10000, hidden=8192]\nGPU: 0-15', 
               shape='parallelogram', style='filled', fillcolor='lightgray')
    
    # Layer 3 - Expert Parallelism
    with dot.subgraph(name='cluster_layer3') as c:
        c.attr(label='Layer 3 - Expert Parallelism', style='dashed')
        
        # MHA for layer 3
        c.node('layer3_mha', 'Multi-Head Attention\n[batch=1024, seq=10000, heads=16, d_k=512]\nGPU: 0-15', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c.node('layer3_residual1', 'Residual Add\n[batch=1024, seq=10000, hidden=8192]\nGPU: 0-15', 
               shape='parallelogram', style='filled', fillcolor='lightgray')
        
        # 16 experts for layer 3 - one per GPU
        for i in range(16):
            c.node(f'layer3_expert{i}', f'Expert {i}\n[batch=variable, seq=variable, hidden=8192->32768->8192]\nGPU: {i}', 
                   shape='rectangle', style='filled', fillcolor='lightyellow')
        
        c.node('layer3_aggregate', 'Aggregate Expert Outputs\n[batch=1024, seq=10000, hidden=8192]\nGPU: 0-15', 
               shape='parallelogram', style='filled', fillcolor='lightcoral')
        c.node('layer3_residual2', 'Residual Add\n[batch=1024, seq=10000, hidden=8192]\nGPU: 0-15', 
               shape='parallelogram', style='filled', fillcolor='lightgray')
    
    # Output
    dot.node('output', 'Output\n[batch_size=1024, seq_len=10000, hidden=8192]', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connections
    dot.edge('input', 'layer0_mha')
    dot.edge('layer0_mha', 'layer0_residual1')
    dot.edge('layer0_residual1', 'gate0')
    
    # Connect gating to experts
    for i in range(16):
        dot.edge('gate0', f'layer0_expert{i}', style='dashed', label=f'route to GPU {i}')
        dot.edge(f'layer0_expert{i}', 'layer0_aggregate')
    
    dot.edge('layer0_aggregate', 'layer0_residual2')
    dot.edge('layer0_residual2', 'layer1_mha')
    dot.edge('layer1_mha', 'layer1_residual1')
    dot.edge('layer1_residual1', 'gate1')
    
    for i in range(16):
        dot.edge('gate1', f'layer1_expert{i}', style='dashed', label=f'route to GPU {i}')
        dot.edge(f'layer1_expert{i}', 'layer1_aggregate')
    
    dot.edge('layer1_aggregate', 'layer1_residual2')
    dot.edge('layer1_residual2', 'layer2_mha')
    dot.edge('layer2_mha', 'layer2_residual1')
    dot.edge('layer2_residual1', 'gate2')
    
    for i in range(16):
        dot.edge('gate2', f'layer2_expert{i}', style='dashed', label=f'route to GPU {i}')
        dot.edge(f'layer2_expert{i}', 'layer2_aggregate')
    
    dot.edge('layer2_aggregate', 'layer2_residual2')
    dot.edge('layer2_residual2', 'layer3_mha')
    dot.edge('layer3_mha', 'layer3_residual1')
    dot.edge('layer3_residual1', 'gate3')
    
    for i in range(16):
        dot.edge('gate3', f'layer3_expert{i}', style='dashed', label=f'route to GPU {i}')
        dot.edge(f'layer3_expert{i}', 'layer3_aggregate')
    
    dot.edge('layer3_aggregate', 'layer3_residual2')
    dot.edge('layer3_residual2', 'output')
    
    # Add residual connections
    dot.edge('input', 'layer0_residual1', style='dashed')
    dot.edge('layer0_residual1', 'layer0_residual2', style='dashed')
    dot.edge('layer0_residual2', 'layer1_residual1', style='dashed')
    dot.edge('layer1_residual1', 'layer1_residual2', style='dashed')
    dot.edge('layer1_residual2', 'layer2_residual1', style='dashed')
    dot.edge('layer2_residual1', 'layer2_residual2', style='dashed')
    dot.edge('layer2_residual2', 'layer3_residual1', style='dashed')
    dot.edge('layer3_residual1', 'layer3_residual2', style='dashed')
    
    return dot

if __name__ == "__main__":
    # Generate baseline DAG
    baseline_dag = create_baseline_dag()
    baseline_dag.render(os.path.join(output_dir, 'baseline_moe_deployment'), format='svg', cleanup=False)
    baseline_dag.save(os.path.join(output_dir, 'baseline_moe_deployment.dot'))
    
    # Generate proposed DAG
    proposed_dag = create_proposed_dag()
    proposed_dag.render(os.path.join(output_dir, 'proposed_cross_node_expert_parallelism'), format='svg', cleanup=False)
    proposed_dag.save(os.path.join(output_dir, 'proposed_cross_node_expert_parallelism.dot'))
    
    print("DAGs generated successfully!")
    print(f"Files saved in: {output_dir}")