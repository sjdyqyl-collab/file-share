#!/usr/bin/env python3

import graphviz

def create_helix_moe_proposed_dag():
    dot = graphviz.Digraph('Helix_MoE_Two_Level_Partitioning', 
                          filename='helix_moe_proposed_dag.svg',
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
    
    # Gate computation (shared across all devices)
    dot.node('gate', 'Gate Network\nCompute routing scores\n[1024×L×4 experts]', 
             shape='rectangle', style='filled', fillcolor='lightgreen')
    dot.edge('broadcast', 'gate')
    
    # Device partitions for MHA (4×4 grid = 16 devices)
    devices = []
    for i in range(4):  # head groups 0-3
        for j in range(4):  # dimension segments 0-3
            device_id = i * 4 + j
            
            # Create partition nodes for MHA
            with dot.subgraph(name=f'cluster_device_{device_id}') as c:
                c.attr(label=f'Device {device_id} (GPU {device_id})', 
                      style='rounded', bgcolor='lightgray', color='black')
                
                # MHA computation
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
    
    # Intra-group concatenation for MHA
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
    
    # Inter-group concatenation for MHA
    mha_concat = 'mha_final_concat'
    dot.node(mha_concat, 'MHA Final Concat\nConcat 4 head groups\n[1024×L×8192]', 
            shape='parallelogram', style='filled', fillcolor='gold')
    
    for i, concat_node in enumerate(intra_concat_nodes):
        dot.edge(concat_node, mha_concat)
    
    # MoE layer - Expert routing
    dot.node('routing', 'Expert Routing\nSelect top-2 experts\n[1024×L×2]', 
            shape='parallelogram', style='filled', fillcolor='purple')
    dot.edge('gate', 'routing', style='dashed')
    dot.edge(mha_concat, 'routing')
    
    # Expert computation (4 experts distributed across devices)
    expert_nodes = []
    for expert_id in range(4):
        expert_node = f'expert_{expert_id}'
        # Map experts to devices (4 experts across 16 devices)
        device_start = expert_id * 4
        device_end = device_start + 3
        
        dot.node(expert_node, f'Expert {expert_id}\nFFN computation\n[1024×L×8192]\nDevices {device_start}-{device_end}', 
                shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Routing connection
        dot.edge('routing', expert_node, style='dashed', label=f'expert_{expert_id}')
        
        expert_nodes.append(expert_node)
    
    # Expert aggregation
    dot.node('expert_agg', 'Expert Aggregation\nWeighted sum of experts\n[1024×L×8192]', 
            shape='parallelogram', style='filled', fillcolor='gold')
    
    for expert_node in expert_nodes:
        dot.edge(expert_node, 'expert_agg')
    
    # Final residual connection
    dot.node('residual', 'Residual Add\n[1024×L×8192]', 
            shape='parallelogram', style='filled', fillcolor='orange')
    dot.edge(mha_concat, 'residual')
    dot.edge('expert_agg', 'residual')
    
    # Output
    dot.node('output', 'Output\n[1024×L×8192]', 
            shape='ellipse', style='filled', fillcolor='lightblue')
    dot.edge('residual', 'output')
    
    # Add legend
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='Legend', style='dashed', bgcolor='white', color='gray')
        c.node('legend_input', 'Input/Output', shape='ellipse', style='filled', fillcolor='lightblue')
        c.node('legend_compute', 'Computation', shape='rectangle', style='filled', fillcolor='lightgreen')
        c.node('legend_comm', 'Communication/Aggregation', shape='parallelogram', style='filled', fillcolor='lightyellow')
        c.node('legend_routing', 'Routing/Gating', shape='parallelogram', style='filled', fillcolor='purple')
        c.node('legend_dashed', 'Routing Selection', style='dashed')
    
    return dot

if __name__ == '__main__':
    dag = create_helix_moe_proposed_dag()
    dag.render()
    print("Generated helix_moe_proposed_dag.svg")