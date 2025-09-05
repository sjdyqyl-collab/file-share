#!/usr/bin/env python3
"""
Generate baseline MoE DAG with TP=8, PP=2, 4 experts per GPU
"""

import graphviz

def generate_baseline_dag():
    dot = graphviz.Digraph('baseline_moe_dag', comment='Baseline MoE Deployment DAG')
    dot.attr(rankdir='TB', size='20,30')
    
    # Input node
    dot.node('input', 'Input\n[1024, 8192]', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Stage 0 (Layers 0-1) - 8 GPUs with TP=8
    with dot.subgraph(name='stage0') as c:
        c.attr(rank='same')
        c.attr(label='Pipeline Stage 0 (Layers 0-1)\nGPUs 0-7 with TP=8')
        
        # Layer 0 components across 8 GPUs
        for gpu_id in range(8):
            gpu_name = f'gpu{gpu_id}'
            
            # Attention components
            c.node(f'attn0_q_{gpu_id}', f'Q Linear\n[1024, 8192]→[1024, 8192]\nGPU {gpu_id}', 
                   shape='rectangle', style='filled', fillcolor='lightgreen')
            c.node(f'attn0_k_{gpu_id}', f'K Linear\n[1024, 8192]→[1024, 8192]\nGPU {gpu_id}', 
                   shape='rectangle', style='filled', fillcolor='lightgreen')
            c.node(f'attn0_v_{gpu_id}', f'V Linear\n[1024, 8192]→[1024, 8192]\nGPU {gpu_id}', 
                   shape='rectangle', style='filled', fillcolor='lightgreen')
            
            c.node(f'attn0_score_{gpu_id}', f'Attention Score\n[1024, 1024]\nGPU {gpu_id}', 
                   shape='rectangle', style='filled', fillcolor='yellow')
            c.node(f'attn0_out_{gpu_id}', f'Attention Output\n[1024, 8192]\nGPU {gpu_id}', 
                   shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # Residual connection
            c.node(f'residual0_{gpu_id}', f'Residual Add\n[1024, 8192]\nGPU {gpu_id}', 
                   shape='parallelogram', style='filled', fillcolor='orange')
            
            # Experts for this GPU (4 experts per GPU)
            for expert_idx in range(4):
                expert_id = gpu_id * 4 + expert_idx
                c.node(f'expert0_{expert_id}', f'Expert {expert_id}\nMLP\n[1024, 32768]→[1024, 8192]\nGPU {gpu_id}', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # Gate
            c.node(f'gate0_{gpu_id}', f'Gate\nSelect Top-K\nGPU {gpu_id}', 
                   shape='parallelogram', style='filled', fillcolor='lightpink')
            
            # Expert aggregation
            c.node(f'expert_agg0_{gpu_id}', f'Expert Aggregation\n[1024, 8192]\nGPU {gpu_id}', 
                   shape='parallelogram', style='filled', fillcolor='lightgray')
            
            # Final residual
            c.node(f'final_residual0_{gpu_id}', f'Final Residual\n[1024, 8192]\nGPU {gpu_id}', 
                   shape='parallelogram', style='filled', fillcolor='orange')
    
    # Stage 1 (Layers 2-3) - 8 GPUs with TP=8
    with dot.subgraph(name='stage1') as c:
        c.attr(rank='same')
        c.attr(label='Pipeline Stage 1 (Layers 2-3)\nGPUs 8-15 with TP=8')
        
        # Layer 2 components across 8 GPUs (using GPUs 8-15)
        for gpu_id in range(8, 16):
            actual_gpu = gpu_id - 8
            
            # Attention components
            c.node(f'attn2_q_{gpu_id}', f'Q Linear\n[1024, 8192]→[1024, 8192]\nGPU {gpu_id}', 
                   shape='rectangle', style='filled', fillcolor='lightgreen')
            c.node(f'attn2_k_{gpu_id}', f'K Linear\n[1024, 8192]→[1024, 8192]\nGPU {gpu_id}', 
                   shape='rectangle', style='filled', fillcolor='lightgreen')
            c.node(f'attn2_v_{gpu_id}', f'V Linear\n[1024, 8192]→[1024, 8192]\nGPU {gpu_id}', 
                   shape='rectangle', style='filled', fillcolor='lightgreen')
            
            c.node(f'attn2_score_{gpu_id}', f'Attention Score\n[1024, 1024]\nGPU {gpu_id}', 
                   shape='rectangle', style='filled', fillcolor='yellow')
            c.node(f'attn2_out_{gpu_id}', f'Attention Output\n[1024, 8192]\nGPU {gpu_id}', 
                   shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # Residual connection
            c.node(f'residual2_{gpu_id}', f'Residual Add\n[1024, 8192]\nGPU {gpu_id}', 
                   shape='parallelogram', style='filled', fillcolor='orange')
            
            # Experts for this GPU (4 experts per GPU)
            for expert_idx in range(4):
                expert_id = (gpu_id - 8) * 4 + expert_idx + 32
                c.node(f'expert2_{expert_id}', f'Expert {expert_id}\nMLP\n[1024, 32768]→[1024, 8192]\nGPU {gpu_id}', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # Gate
            c.node(f'gate2_{gpu_id}', f'Gate\nSelect Top-K\nGPU {gpu_id}', 
                   shape='parallelogram', style='filled', fillcolor='lightpink')
            
            # Expert aggregation
            c.node(f'expert_agg2_{gpu_id}', f'Expert Aggregation\n[1024, 8192]\nGPU {gpu_id}', 
                   shape='parallelogram', style='filled', fillcolor='lightgray')
            
            # Final residual
            c.node(f'final_residual2_{gpu_id}', f'Final Residual\n[1024, 8192]\nGPU {gpu_id}', 
                   shape='parallelogram', style='filled', fillcolor='orange')
    
    # Output node
    dot.node('output', 'Output\n[1024, 8192]', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connections for Stage 0, Layer 0
    for gpu_id in range(8):
        # Input to attention
        dot.edge('input', f'attn0_q_{gpu_id}')
        dot.edge('input', f'attn0_k_{gpu_id}')
        dot.edge('input', f'attn0_v_{gpu_id}')
        
        # Attention computation
        dot.edge(f'attn0_q_{gpu_id}', f'attn0_score_{gpu_id}')
        dot.edge(f'attn0_k_{gpu_id}', f'attn0_score_{gpu_id}')
        dot.edge(f'attn0_v_{gpu_id}', f'attn0_out_{gpu_id}')
        dot.edge(f'attn0_score_{gpu_id}', f'attn0_out_{gpu_id}')
        
        # Residual connection
        dot.edge('input', f'residual0_{gpu_id}')
        dot.edge(f'attn0_out_{gpu_id}', f'residual0_{gpu_id}')
        
        # Gate and experts
        dot.edge(f'residual0_{gpu_id}', f'gate0_{gpu_id}')
        
        # Expert selection (dashed lines for gate connections)
        for expert_idx in range(4):
            expert_id = gpu_id * 4 + expert_idx
            dot.edge(f'gate0_{gpu_id}', f'expert0_{expert_id}', style='dashed')
            dot.edge(f'residual0_{gpu_id}', f'expert0_{expert_id}')
            dot.edge(f'expert0_{expert_id}', f'expert_agg0_{gpu_id}')
        
        # Expert aggregation to final residual
        dot.edge(f'expert_agg0_{gpu_id}', f'final_residual0_{gpu_id}')
        dot.edge(f'residual0_{gpu_id}', f'final_residual0_{gpu_id}')
    
    # Pipeline communication between stages
    dot.edge('final_residual0_7', 'attn2_q_8', label='Pipeline Comm\nStage 0→1', style='dotted')
    
    # Connections for Stage 1, Layer 2
    for gpu_id in range(8, 16):
        # Similar connections as layer 0
        dot.edge(f'final_residual0_7', f'attn2_q_{gpu_id}')
        dot.edge(f'final_residual0_7', f'attn2_k_{gpu_id}')
        dot.edge(f'final_residual0_7', f'attn2_v_{gpu_id}')
        
        # Attention computation
        dot.edge(f'attn2_q_{gpu_id}', f'attn2_score_{gpu_id}')
        dot.edge(f'attn2_k_{gpu_id}', f'attn2_score_{gpu_id}')
        dot.edge(f'attn2_v_{gpu_id}', f'attn2_out_{gpu_id}')
        dot.edge(f'attn2_score_{gpu_id}', f'attn2_out_{gpu_id}')
        
        # Residual connection
        dot.edge(f'final_residual0_7', f'residual2_{gpu_id}')
        dot.edge(f'attn2_out_{gpu_id}', f'residual2_{gpu_id}')
        
        # Gate and experts
        dot.edge(f'residual2_{gpu_id}', f'gate2_{gpu_id}')
        
        # Expert selection
        for expert_idx in range(4):
            expert_id = (gpu_id - 8) * 4 + expert_idx + 32
            dot.edge(f'gate2_{gpu_id}', f'expert2_{expert_id}', style='dashed')
            dot.edge(f'residual2_{gpu_id}', f'expert2_{expert_id}')
            dot.edge(f'expert2_{expert_id}', f'expert_agg2_{gpu_id}')
        
        # Expert aggregation to final residual
        dot.edge(f'expert_agg2_{gpu_id}', f'final_residual2_{gpu_id}')
        dot.edge(f'residual2_{gpu_id}', f'final_residual2_{gpu_id}')
        
        # Connect to output
        dot.edge(f'final_residual2_{gpu_id}', 'output')
    
    return dot

if __name__ == '__main__':
    dag = generate_baseline_dag()
    dag.render('/home/wzc/data/file-share/2025-09-04-17-37-41/baseline_moe_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-04-17-37-41/baseline_moe_dag.dot')
    print("Baseline DAG generated successfully")