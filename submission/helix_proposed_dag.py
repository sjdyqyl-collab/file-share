#!/usr/bin/env python3

import graphviz

def create_helix_proposed_dag():
    dot = graphviz.Digraph('Helix_Two_Level_Partitioning', 
                          filename='helix_proposed_dag.svg',
                          format='svg',
                          graph_attr={'rankdir': 'TB', 'bgcolor': 'white'},
                          node_attr={'fontname': 'Arial', 'fontsize': '10'})
    
    # Input node
    dot.node('input', 'Input\nX: [B×L×8192]\n(B=1024, L=seq_len)', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Broadcast input to all devices
    dot.node('broadcast', 'Broadcast\nX to all GPUs', 
             shape='parallelogram', style='filled', fillcolor='lightyellow')
    dot.edge('input', 'broadcast')
    
    # Device partitions (4×4 grid = 16 devices)
    devices = []
    for i in range(4):  # head groups 0-3
        for j in range(4):  # dimension segments 0-3
            device_id = i * 4 + j
            
            # Create partition nodes
            with dot.subgraph(name=f'cluster_device_{device_id}') as c:
                c.attr(label=f'Device {device_id} (GPU {device_id})', 
                      style='rounded', bgcolor='lightgray', color='black')
                
                # QKV computation
                q_node = f'q_{i}_{j}'
                k_node = f'k_{i}_{j}'
                v_node = f'v_{i}_{j}'
                
                c.node(q_node, f'Q_{i},{j} = XW_Q^{i,j}\n[1024×L×128×4]', 
                      shape='rectangle', style='filled', fillcolor='lightgreen')
                c.node(k_node, f'K_{i},{j} = XW_K^{i,j}\n[1024×L×128×4]', 
                      shape='rectangle', style='filled', fillcolor='lightgreen')
                c.node(v_node, f'V_{i},{j} = XW_V^{i,j}\n[1024×L×128×4]', 
                      shape='rectangle', style='filled', fillcolor='lightgreen')
                
                # Attention computation
                attn_node = f'attn_{i}_{j}'
                c.node(attn_node, f'Attention_{i},{j}\n= softmax(QK^T/√128)V\n[1024×L×128×4]', 
                      shape='rectangle', style='filled', fillcolor='lightcoral')
                
                # Connect within device
                c.edge(q_node, attn_node)
                c.edge(k_node, attn_node)
                c.edge(v_node, attn_node)
                
                # Connect broadcast to QKV
                dot.edge('broadcast', q_node)
                dot.edge('broadcast', k_node)
                dot.edge('broadcast', v_node)
                
                devices.append((device_id, i, j, attn_node))
    
    # Intra-group concatenation (concatenate dimension segments within each head group)
    intra_concat_nodes = []
    for i in range(4):  # head groups
        concat_node = f'intra_concat_{i}'
        dot.node(concat_node, f'Intra-Group Concat {i}\nConcat 4×128 dims\n[1024×L×512×4 heads]', 
                shape='parallelogram', style='filled', fillcolor='orange')
        intra_concat_nodes.append(concat_node)
        
        # Connect all devices in this head group
        for j in range(4):  # dimension segments
            device_id = i * 4 + j
            dot.edge(f'attn_{i}_{j}', concat_node)
    
    # Inter-group concatenation (concatenate head groups)
    final_concat = 'final_concat'
    dot.node(final_concat, 'Final Concat\nConcat 4 head groups\n[1024×L×8192]', 
            shape='parallelogram', style='filled', fillcolor='gold')
    
    for i, concat_node in enumerate(intra_concat_nodes):
        dot.edge(concat_node, final_concat)
    
    # Output
    dot.node('output', 'Output\n[1024×L×8192]', 
            shape='ellipse', style='filled', fillcolor='lightblue')
    dot.edge(final_concat, 'output')
    
    # Add legend
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='Legend', style='dashed', bgcolor='white', color='gray')
        c.node('legend_input', 'Input/Output', shape='ellipse', style='filled', fillcolor='lightblue')
        c.node('legend_compute', 'Computation', shape='rectangle', style='filled', fillcolor='lightgreen')
        c.node('legend_comm', 'Communication/Aggregation', shape='parallelogram', style='filled', fillcolor='lightyellow')
    
    return dot

if __name__ == '__main__':
    dag = create_helix_proposed_dag()
    dag.render()
    print("Generated helix_proposed_dag.svg")