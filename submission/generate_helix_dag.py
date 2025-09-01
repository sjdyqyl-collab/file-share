#!/usr/bin/env python3
"""
Generate DAG for Helix: Two-Level Attention Partitioning
"""

import graphviz

def create_helix_dag():
    dot = graphviz.Digraph('helix_two_level_partitioning', 
                          comment='Helix: Two-Level Attention Partitioning DAG',
                          node_attr={'shape': 'rectangle'})
    
    dot.attr(rankdir='TB', size='20,30', fontname='Arial')
    
    # Input layer
    dot.node('input', 'Input Tensor\nX: [1024, L, 8192]\nGPU: all', 
             shape='parallelogram', style='filled', fillcolor='lightblue')
    
    # Level 1: Head Group Distribution
    dot.node('head_split', 'Head Group Split\n16 heads → 4 groups\nGPU: all', 
             shape='ellipse', style='filled', fillcolor='lightyellow')
    
    # Level 2: Dimension Partitioning
    dot.node('dim_split', 'Dimension Slice\n512 dims → 4 slices\nGPU: all', 
             shape='ellipse', style='filled', fillcolor='lightyellow')
    
    # Device assignments (4x4 grid = 16 devices)
    devices = []
    for i in range(4):  # head groups
        for j in range(4):  # dimension slices
            device_id = i * 4 + j
            devices.append(f'device_{device_id}')
            
            # Input distribution
            dot.node(f'input_{device_id}', 
                     f'Input Slice\nX[{i},{j}]: [1024, L, 512]\nGPU: {device_id}',
                     shape='parallelogram', style='filled', fillcolor='lightgreen')
            
            # Q projection
            dot.node(f'q_proj_{device_id}', 
                     f'Q Projection\nW_Q[{i},{j}]: [2048, 2048]\nInput: [1024, L, 512]\nOutput: [1024, L, 128]\nGPU: {device_id}',
                     shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # K projection
            dot.node(f'k_proj_{device_id}', 
                     f'K Projection\nW_K[{i},{j}]: [2048, 2048]\nInput: [1024, L, 512]\nOutput: [1024, L, 128]\nGPU: {device_id}',
                     shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # V projection
            dot.node(f'v_proj_{device_id}', 
                     f'V Projection\nW_V[{i},{j}]: [2048, 2048]\nInput: [1024, L, 512]\nOutput: [1024, L, 128]\nGPU: {device_id}',
                     shape='rectangle', style='filled', fillcolor='lightcoral')
            
            # Attention computation
            dot.node(f'attn_{device_id}', 
                     f'Scaled Dot-Product Attention\nQ,K,V: [1024, L, 128]\nScale: 1/√128\nOutput: [1024, L, 128]\nGPU: {device_id}',
                     shape='rectangle', style='filled', fillcolor='lightpink')
            
            # Connections for each device
            dot.edge('input', 'head_split')
            dot.edge('head_split', 'dim_split')
            dot.edge('dim_split', f'input_{device_id}')
            dot.edge(f'input_{device_id}', f'q_proj_{device_id}')
            dot.edge(f'input_{device_id}', f'k_proj_{device_id}')
            dot.edge(f'input_{device_id}', f'v_proj_{device_id}')
            dot.edge(f'q_proj_{device_id}', f'attn_{device_id}')
            dot.edge(f'k_proj_{device_id}', f'attn_{device_id}')
            dot.edge(f'v_proj_{device_id}', f'attn_{device_id}')
    
    # Intra-group concatenation (4 groups, 4 devices each)
    for group_id in range(4):
        group_devices = [group_id * 4 + j for j in range(4)]
        
        dot.node(f'intra_concat_{group_id}', 
                 f'Intra-Group Concat\nGroup {group_id}\nConcat 4×[1024,L,128] → [1024,L,512]\nGPU: {group_devices}',
                 shape='ellipse', style='filled', fillcolor='orange')
        
        for j in range(4):
            device_id = group_id * 4 + j
            dot.edge(f'attn_{device_id}', f'intra_concat_{group_id}')
    
    # Inter-group concatenation
    dot.node('inter_concat', 
             'Inter-Group Concat\nConcat 4×[1024,L,512] → [1024,L,8192]\nGPU: all',
             shape='ellipse', style='filled', fillcolor='gold')
    
    for group_id in range(4):
        dot.edge(f'intra_concat_{group_id}', 'inter_concat')
    
    # Output
    dot.node('output', 'Output Tensor\n[1024, L, 8192]\nGPU: all',
             shape='parallelogram', style='filled', fillcolor='lightblue')
    
    dot.edge('inter_concat', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_helix_dag()
    dag.render('/home/wzc/data/file-share/submission/helix_two_level_partitioning', format='svg', cleanup=True)
    dag.save('/home/wzc/data/file-share/submission/helix_two_level_partitioning.dot')