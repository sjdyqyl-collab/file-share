#!/usr/bin/env python3

import graphviz

def create_helix_dag():
    """Create a complete DAG for Helix two-level attention partitioning on 16 GPUs"""
    
    # Create directed graph
    dot = graphviz.Digraph('Helix_Two_Level_Attention_Partitioning', 
                          comment='Helix Model Deployment on 16 GPUs')
    
    # Set graph attributes
    dot.attr(rankdir='TB', size='20,30', compound='true', 
             bgcolor='white', fontname='Arial')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fontname='Arial')
    dot.attr('edge', fontname='Arial')
    
    # Level 0: Input
    dot.node('input', 'Input Tensor\nX ∈ ℝ^(1024×L×8192)', 
             fillcolor='lightblue', shape='parallelogram')
    
    # Level 1: Input Broadcast
    dot.node('broadcast', 'Broadcast\nAll 16 GPUs', 
             fillcolor='lightgreen', shape='ellipse')
    dot.edge('input', 'broadcast', label='Full tensor')
    
    # Create subgraphs for each GPU
    for gpu_id in range(16):
        with dot.subgraph(name=f'cluster_gpu{gpu_id}') as c:
            # Calculate group and slice indices
            group_id = gpu_id // 4  # 0-3
            slice_id = gpu_id % 4   # 0-3
            
            c.attr(label=f'GPU {gpu_id}\nGroup {group_id}, Slice {slice_id}', 
                   style='rounded', bgcolor='lightgray', color='black')
            
            # Projection nodes
            c.node(f'proj_q_{gpu_id}', 
                   f'Q Projection\nW_Q^({group_id},{slice_id})\nℝ^(8192×128)', 
                   fillcolor='lightyellow')
            c.node(f'proj_k_{gpu_id}', 
                   f'K Projection\nW_K^({group_id},{slice_id})\nℝ^(8192×128)', 
                   fillcolor='lightyellow')
            c.node(f'proj_v_{gpu_id}', 
                   f'V Projection\nW_V^({group_id},{slice_id})\nℝ^(8192×128)', 
                   fillcolor='lightyellow')
            
            # Intermediate tensors
            c.node(f'q_{gpu_id}', f'Q^({group_id},{slice_id})\nℝ^(1024×L×128)', 
                   fillcolor='lightcyan')
            c.node(f'k_{gpu_id}', f'K^({group_id},{slice_id})\nℝ^(1024×L×128)', 
                   fillcolor='lightcyan')
            c.node(f'v_{gpu_id}', f'V^({group_id},{slice_id})\nℝ^(1024×L×128)', 
                   fillcolor='lightcyan')
            
            # Attention computation
            c.node(f'attn_{gpu_id}', 
                   f'Softmax Attention\nQK^TV/√128\nℝ^(1024×L×128)', 
                   fillcolor='lightpink')
            c.node(f'out_{gpu_id}', f'Attention^({group_id},{slice_id})\nℝ^(1024×L×128)', 
                   fillcolor='lightgreen')
            
            # Edges within GPU
            c.edge('broadcast', f'proj_q_{gpu_id}')
            c.edge('broadcast', f'proj_k_{gpu_id}')
            c.edge('broadcast', f'proj_v_{gpu_id}')
            c.edge(f'proj_q_{gpu_id}', f'q_{gpu_id}')
            c.edge(f'proj_k_{gpu_id}', f'k_{gpu_id}')
            c.edge(f'proj_v_{gpu_id}', f'v_{gpu_id}')
            c.edge(f'q_{gpu_id}', f'attn_{gpu_id}')
            c.edge(f'k_{gpu_id}', f'attn_{gpu_id}')
            c.edge(f'v_{gpu_id}', f'attn_{gpu_id}')
            c.edge(f'attn_{gpu_id}', f'out_{gpu_id}')
    
    # Level 4: Intra-group concatenation
    for group_id in range(4):
        start_gpu = group_id * 4
        end_gpu = start_gpu + 3
        
        concat_node = f'concat_group_{group_id}'
        dot.node(concat_node, 
                 f'Intra-group Concat\nGroup {group_id}\nℝ^(1024×L×512)', 
                 fillcolor='orange', shape='diamond')
        
        # Connect all GPUs in group to concatenation node
        for gpu_id in range(start_gpu, start_gpu + 4):
            dot.edge(f'out_{gpu_id}', concat_node, 
                     label=f'GPU {gpu_id} slice', style='dashed')
    
    # Level 5: Inter-group concatenation
    dot.node('final_concat', 'Final Concatenation\nℝ^(1024×L×8192)', 
             fillcolor='gold', shape='diamond')
    
    for group_id in range(4):
        dot.edge(f'concat_group_{group_id}', 'final_concat', 
                 label=f'Group {group_id} output', style='bold')
    
    # Output
    dot.node('output', 'Output Tensor\nℝ^(1024×L×8192)', 
             fillcolor='lightblue', shape='parallelogram')
    dot.edge('final_concat', 'output')
    
    # Add communication edges with labels
    for group_id in range(4):
        start_gpu = group_id * 4
        
        # All-gather communication within groups
        comm_node = f'comm_group_{group_id}'
        dot.node(comm_node, f'All-gather\nGroup {group_id}', 
                 fillcolor='red', shape='ellipse', style='dashed')
        
        # Connect communication to concatenation
        dot.edge(comm_node, f'concat_group_{group_id}', 
                 label='Gather 4 slices', style='dashed')
    
    # Final communication
    dot.node('final_comm', 'Final Gather\nAcross Groups', 
             fillcolor='red', shape='ellipse', style='dashed')
    dot.edge('final_comm', 'final_concat', 
             label='Gather 4 group outputs', style='dashed')
    
    return dot

if __name__ == '__main__':
    dag = create_helix_dag()
    
    # Save as SVG
    dag.render('/home/wzc/data/file-share/submission/helix_dag', 
               format='svg', cleanup=True)
    
    # Also save the source code
    with open('/home/wzc/data/file-share/submission/helix_dag.dot', 'w') as f:
        f.write(dag.source)
    
    print("Helix DAG generated successfully!")
    print("SVG file: /home/wzc/data/file-share/submission/helix_dag.svg")
    print("DOT file: /home/wzc/data/file-share/submission/helix_dag.dot")