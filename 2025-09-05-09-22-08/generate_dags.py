#!/usr/bin/env python3
"""
Generate DAGs for MoE model deployments based on the paper
"Large-Scale Cross-Node Expert Parallelism for Mixture-of-Experts Models"
"""

import os
from graphviz import Digraph

def create_baseline_dag():
    """Create baseline DAG with TP=8, PP=2, 16 GPUs total, 4 experts per GPU"""
    
    dot = Digraph(name='baseline_moe_dag', 
                  comment='Baseline MoE Deployment: TP=8, PP=2, 16 GPUs, 4 experts/GPU')
    dot.attr(rankdir='TB', size='20,30')
    
    # Global attributes
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Input
    dot.node('input', 'Input\n[1024, 2048, 8192]', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Pipeline Stage 0 (GPUs 0-7)
    with dot.subgraph(name='cluster_stage0') as c:
        c.attr(label='Pipeline Stage 0\nGPUs 0-7', style='rounded', fillcolor='lightgray')
        
        # Layer 0 for each GPU in stage 0
        for gpu_id in range(8):
            gpu_name = f'gpu0_{gpu_id}'
            
            # Attention for GPU
            c.node(f'attn0_{gpu_id}', f'Attention\nGPU{gpu_id}\n[1024, 2048, 8192]\nTP-Slice', 
                   shape='rectangle', style='filled', fillcolor='lightyellow')
            
            # Gate for GPU
            c.node(f'gate0_{gpu_id}', f'Gate\nGPU{gpu_id}\n[1024, 2048, 16]', 
                   shape='parallelogram', style='filled', fillcolor='lightgreen')
            
            # 4 Experts per GPU
            for expert_idx in range(4):
                expert_id = gpu_id * 4 + expert_idx
                c.node(f'expert0_{gpu_id}_{expert_idx}', 
                       f'Expert{expert_id}\nGPU{gpu_id}\n[1024, 2048, 32768]\nColocated', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # Add nodes for expert aggregation
            c.node(f'agg0_{gpu_id}', f'Expert Agg\nGPU{gpu_id}\n[1024, 2048, 8192]', 
                   shape='parallelogram', style='filled', fillcolor='lightblue')
            
            # Residual add
            c.node(f'residual0_{gpu_id}', f'Residual Add\nGPU{gpu_id}\n[1024, 2048, 8192]', 
                   shape='ellipse', style='filled', fillcolor='lightpink')
    
    # Pipeline Stage 1 (GPUs 8-15)
    with dot.subgraph(name='cluster_stage1') as c:
        c.attr(label='Pipeline Stage 1\nGPUs 8-15', style='rounded', fillcolor='lightgray')
        
        # Layer 1 for each GPU in stage 1
        for gpu_id in range(8, 16):
            actual_gpu = gpu_id - 8
            
            # Attention for GPU
            c.node(f'attn1_{gpu_id}', f'Attention\nGPU{gpu_id}\n[1024, 2048, 8192]\nTP-Slice', 
                   shape='rectangle', style='filled', fillcolor='lightyellow')
            
            # Gate for GPU
            c.node(f'gate1_{gpu_id}', f'Gate\nGPU{gpu_id}\n[1024, 2048, 16]', 
                   shape='parallelogram', style='filled', fillcolor='lightgreen')
            
            # 4 Experts per GPU
            for expert_idx in range(4):
                expert_id = (gpu_id - 8) * 4 + expert_idx + 32  # Offset for layer 1
                c.node(f'expert1_{gpu_id}_{expert_idx}', 
                       f'Expert{expert_id}\nGPU{gpu_id}\n[1024, 2048, 32768]\nColocated', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # Add nodes for expert aggregation
            c.node(f'agg1_{gpu_id}', f'Expert Agg\nGPU{gpu_id}\n[1024, 2048, 8192]', 
                   shape='parallelogram', style='filled', fillcolor='lightblue')
            
            # Residual add
            c.node(f'residual1_{gpu_id}', f'Residual Add\nGPU{gpu_id}\n[1024, 2048, 8192]', 
                   shape='ellipse', style='filled', fillcolor='lightpink')
    
    # Output
    dot.node('output', 'Output\n[1024, 2048, 8192]', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connections for Stage 0
    for gpu_id in range(8):
        # Input to attention
        dot.edge('input', f'attn0_{gpu_id}', label='Broadcast')
        
        # Attention to gate
        dot.edge(f'attn0_{gpu_id}', f'gate0_{gpu_id}')
        
        # Gate to experts (dashed for routing)
        for expert_idx in range(4):
            dot.edge(f'gate0_{gpu_id}', f'expert0_{gpu_id}_{expert_idx}', 
                    style='dashed', label=f'Route tokens')
        
        # Experts to aggregation
        for expert_idx in range(4):
            dot.edge(f'expert0_{gpu_id}_{expert_idx}', f'agg0_{gpu_id}')
        
        # Aggregation to residual
        dot.edge(f'agg0_{gpu_id}', f'residual0_{gpu_id}')
        dot.edge(f'attn0_{gpu_id}', f'residual0_{gpu_id}')  # Skip connection
        
        # Residual to pipeline stage 1
        dot.edge(f'residual0_{gpu_id}', f'attn1_{gpu_id+8}', label='Pipeline send')
    
    # Connections for Stage 1
    for gpu_id in range(8, 16):
        # Attention to gate
        dot.edge(f'attn1_{gpu_id}', f'gate1_{gpu_id}')
        
        # Gate to experts (dashed for routing)
        for expert_idx in range(4):
            dot.edge(f'gate1_{gpu_id}', f'expert1_{gpu_id}_{expert_idx}', 
                    style='dashed', label=f'Route tokens')
        
        # Experts to aggregation
        for expert_idx in range(4):
            dot.edge(f'expert1_{gpu_id}_{expert_idx}', f'agg1_{gpu_id}')
        
        # Aggregation to residual
        dot.edge(f'agg1_{gpu_id}', f'residual1_{gpu_id}')
        dot.edge(f'attn1_{gpu_id}', f'residual1_{gpu_id}')  # Skip connection
        
        # Residual to output
        dot.edge(f'residual1_{gpu_id}', 'output', label='Gather')
    
    # Add tensor parallelism connections (within each GPU)
    for stage in [0, 1]:
        stage_offset = stage * 8
        for gpu_id in range(8):
            actual_gpu = stage_offset + gpu_id
            
            # Tensor parallelism within attention
            dot.node(f'tp_attn{stage}_{gpu_id}', f'TP All-Reduce\nAttention\nGPU{actual_gpu}', 
                    shape='ellipse', style='dotted', fillcolor='gray')
            dot.edge(f'attn{stage}_{actual_gpu}', f'tp_attn{stage}_{gpu_id}')
            dot.edge(f'tp_attn{stage}_{gpu_id}', f'gate{stage}_{actual_gpu}')
    
    return dot

def create_proposed_dag():
    """Create proposed DAG with EP=64, one expert per GPU, 64 GPUs total"""
    
    dot = Digraph(name='proposed_moe_dag', 
                  comment='Proposed MoE Deployment: EP=64, 64 GPUs, 1 expert/GPU')
    dot.attr(rankdir='TB', size='30,40')
    
    # Global attributes
    dot.attr('node', fontname='Arial', fontsize='9')
    dot.attr('edge', fontname='Arial', fontsize='7')
    
    # Input
    dot.node('input', 'Input\n[1024, 2048, 8192]', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Create 4 layers with 16 experts each
    for layer in range(4):
        layer_name = f'layer{layer}'
        
        with dot.subgraph(name=f'cluster_layer{layer}') as c:
            c.attr(label=f'Layer {layer}\n16 Experts across 16 GPUs', style='rounded', fillcolor='lightgray')
            
            # Attention (replicated across the layer)
            c.node(f'attn_{layer}', f'Attention\nLayer{layer}\n[1024, 2048, 8192]\nReplicated', 
                   shape='rectangle', style='filled', fillcolor='lightyellow')
            
            # Gate (replicated)
            c.node(f'gate_{layer}', f'Gate\nLayer{layer}\n[1024, 2048, 16]\nReplicated', 
                   shape='parallelogram', style='filled', fillcolor='lightgreen')
            
            # Token router
            c.node(f'router_{layer}', f'Token Router\nLayer{layer}\nAsync Routing', 
                   shape='parallelogram', style='filled', fillcolor='orange')
            
            # 16 experts (one per GPU)
            for expert_id in range(16):
                gpu_id = layer * 16 + expert_id
                node_id = layer // 2  # 2 layers per node
                gpu_in_node = expert_id
                
                c.node(f'expert_{layer}_{expert_id}', 
                       f'Expert{layer*16+expert_id}\nNode{node_id}GPU{gpu_in_node}\n[1024, 2048, 32768]', 
                       shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # Expert aggregation
            c.node(f'agg_{layer}', f'Expert Aggregation\nLayer{layer}\n[1024, 2048, 8192]', 
                   shape='parallelogram', style='filled', fillcolor='lightblue')
            
            # Residual add
            c.node(f'residual_{layer}', f'Residual Add\nLayer{layer}\n[1024, 2048, 8192]', 
                   shape='ellipse', style='filled', fillcolor='lightpink')
    
    # Output
    dot.node('output', 'Output\n[1024, 2048, 8192]', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Connections
    for layer in range(4):
        if layer == 0:
            # Input to attention
            dot.edge('input', f'attn_{layer}')
        else:
            # Previous layer to current layer
            dot.edge(f'residual_{layer-1}', f'attn_{layer}')
        
        # Attention to gate
        dot.edge(f'attn_{layer}', f'gate_{layer}')
        
        # Gate to router
        dot.edge(f'gate_{layer}', f'router_{layer}')
        
        # Router to experts (cross-node communication)
        for expert_id in range(16):
            node_id = layer // 2
            gpu_in_node = expert_id
            dot.edge(f'router_{layer}', f'expert_{layer}_{expert_id}', 
                    style='dashed', 
                    label=f'Node{node_id}GPU{gpu_in_node}')
        
        # Experts to aggregation (gather from all nodes)
        for expert_id in range(16):
            dot.edge(f'expert_{layer}_{expert_id}', f'agg_{layer}')
        
        # Aggregation to residual
        dot.edge(f'agg_{layer}', f'residual_{layer}')
        dot.edge(f'attn_{layer}', f'residual_{layer}')  # Skip connection
    
    # Final output
    dot.edge(f'residual_3', 'output')
    
    # Add communication nodes for clarity
    for layer in range(4):
        # Cross-node communication
        dot.node(f'comm_{layer}', f'Cross-Node\nCommunication\nLayer{layer}', 
                shape='ellipse', style='dotted', fillcolor='gray')
        
        # Connect router to communication
        dot.edge(f'router_{layer}', f'comm_{layer}')
        
        # Connect communication to experts
        for expert_id in range(16):
            dot.edge(f'comm_{layer}', f'expert_{layer}_{expert_id}')
    
    return dot

def main():
    """Generate both DAGs and save them"""
    
    # Create output directory
    output_dir = "/home/wzc/data/file-share/2025-09-05-09-22-08"
    
    # Generate baseline DAG
    print("Generating baseline DAG...")
    baseline_dag = create_baseline_dag()
    
    # Save baseline DOT file
    baseline_dot_path = os.path.join(output_dir, "baseline_moe_dag.dot")
    with open(baseline_dot_path, 'w') as f:
        f.write(baseline_dag.source)
    
    # Save baseline SVG
    baseline_svg_path = os.path.join(output_dir, "baseline_moe_dag.svg")
    baseline_dag.render(baseline_svg_path.replace('.svg', ''), format='svg', cleanup=True)
    
    # Generate proposed DAG
    print("Generating proposed DAG...")
    proposed_dag = create_proposed_dag()
    
    # Save proposed DOT file
    proposed_dot_path = os.path.join(output_dir, "proposed_moe_dag.dot")
    with open(proposed_dot_path, 'w') as f:
        f.write(proposed_dag.source)
    
    # Save proposed SVG
    proposed_svg_path = os.path.join(output_dir, "proposed_moe_dag.svg")
    proposed_dag.render(proposed_svg_path.replace('.svg', ''), format='svg', cleanup=True)
    
    print(f"Generated files:")
    print(f"- {baseline_dot_path}")
    print(f"- {baseline_svg_path}")
    print(f"- {proposed_dot_path}")
    print(f"- {proposed_svg_path}")
    
    return {
        "baseline_dot": baseline_dot_path,
        "baseline_svg": baseline_svg_path,
        "proposed_dot": proposed_dot_path,
        "proposed_svg": proposed_svg_path
    }

if __name__ == "__main__":
    main()