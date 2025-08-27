#!/usr/bin/env python3

import graphviz

def create_helix_moe_baseline_dag():
    dot = graphviz.Digraph('Helix_MoE_Baseline_TP_PP', 
                          filename='helix_moe_baseline_dag.svg',
                          format='svg',
                          graph_attr={'rankdir': 'TB', 'bgcolor': 'white'},
                          node_attr={'fontname': 'Arial', 'fontsize': '10'})
    
    # Input node
    dot.node('input', 'Input\nX: [B×L×8192]\n(B=1024, L=seq_len)', 
             shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Pipeline stage 0 (devices 0-7)
    with dot.subgraph(name='cluster_pipeline_stage_0') as c0:
        c0.attr(label='Pipeline Stage 0\nDevices 0-7 (Tensor Parallel)', 
               style='rounded', bgcolor='lightcyan', color='black')
        
        # Layer 0
        c0.node('layer0_input', 'Layer 0 Input\n[1024×L×8192]', 
               shape='ellipse', style='filled', fillcolor='lightblue')
        
        # MHA for Layer 0
        c0.node('mha0_q', 'MHA Layer 0 - Q\nTensor Parallel Split\n[1024×L×8192] across 8 GPUs', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c0.node('mha0_k', 'MHA Layer 0 - K\nTensor Parallel Split\n[1024×L×8192] across 8 GPUs', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c0.node('mha0_v', 'MHA Layer 0 - V\nTensor Parallel Split\n[1024×L×8192] across 8 GPUs', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        
        c0.node('mha0_attn', 'MHA Layer 0 Attention\nAll-reduce across 8 GPUs\n[1024×L×8192]', 
               shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Gate network for MoE
        c0.node('gate0', 'Gate Network Layer 0\nCompute routing scores\n[1024×L×4 experts]', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Expert computation for Layer 0
        expert0_nodes = []
        for expert_id in range(4):
            expert_node = f'expert0_{expert_id}'
            # Experts distributed within stage 0 (2 experts per 4 devices)
            device_start = expert_id * 2
            device_end = device_start + 1
            
            c0.node(expert_node, f'Expert {expert_id} Layer 0\nFFN computation\n[1024×L×8192]\nDevices {device_start}-{device_end}', 
                   shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # Routing connection
            c0.edge('gate0', expert_node, style='dashed')
            expert0_nodes.append(expert_node)
        
        # Expert aggregation for Layer 0
        c0.node('expert0_agg', 'Expert 0 Aggregation\nWeighted sum\n[1024×L×8192]', 
               shape='parallelogram', style='filled', fillcolor='gold')
        
        for expert_node in expert0_nodes:
            c0.edge(expert_node, 'expert0_agg')
        
        # Residual connection
        c0.node('res0', 'Residual Add Layer 0\n[1024×L×8192]', 
               shape='parallelogram', style='filled', fillcolor='orange')
        
        # Connect within stage 0
        c0.edge('layer0_input', 'mha0_q')
        c0.edge('layer0_input', 'mha0_k')
        c0.edge('layer0_input', 'mha0_v')
        c0.edge('mha0_q', 'mha0_attn')
        c0.edge('mha0_k', 'mha0_attn')
        c0.edge('mha0_v', 'mha0_attn')
        c0.edge('mha0_attn', 'gate0')
        c0.edge('mha0_attn', 'res0')
        c0.edge('expert0_agg', 'res0')
    
    # Pipeline stage 1 (devices 8-15)
    with dot.subgraph(name='cluster_pipeline_stage_1') as c1:
        c1.attr(label='Pipeline Stage 1\nDevices 8-15 (Tensor Parallel)', 
               style='rounded', bgcolor='lightyellow', color='black')
        
        # Layer 1
        c1.node('layer1_input', 'Layer 1 Input\n[1024×L×8192]', 
               shape='ellipse', style='filled', fillcolor='lightblue')
        
        # MHA for Layer 1
        c1.node('mha1_q', 'MHA Layer 1 - Q\nTensor Parallel Split\n[1024×L×8192] across 8 GPUs', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c1.node('mha1_k', 'MHA Layer 1 - K\nTensor Parallel Split\n[1024×L×8192] across 8 GPUs', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c1.node('mha1_v', 'MHA Layer 1 - V\nTensor Parallel Split\n[1024×L×8192] across 8 GPUs', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        
        c1.node('mha1_attn', 'MHA Layer 1 Attention\nAll-reduce across 8 GPUs\n[1024×L×8192]', 
               shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # Gate network for MoE
        c1.node('gate1', 'Gate Network Layer 1\nCompute routing scores\n[1024×L×4 experts]', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        
        # Expert computation for Layer 1
        expert1_nodes = []
        for expert_id in range(4):
            expert_node = f'expert1_{expert_id}'
            # Experts distributed within stage 1 (2 experts per 4 devices)
            device_start = 8 + expert_id * 2
            device_end = device_start + 1
            
            c1.node(expert_node, f'Expert {expert_id} Layer 1\nFFN computation\n[1024×L×8192]\nDevices {device_start}-{device_end}', 
                   shape='rectangle', style='filled', fillcolor='lightgreen')
            
            # Routing connection
            c1.edge('gate1', expert_node, style='dashed')
            expert1_nodes.append(expert_node)
        
        # Expert aggregation for Layer 1
        c1.node('expert1_agg', 'Expert 1 Aggregation\nWeighted sum\n[1024×L×8192]', 
               shape='parallelogram', style='filled', fillcolor='gold')
        
        for expert_node in expert1_nodes:
            c1.edge(expert_node, 'expert1_agg')
        
        # Residual connection
        c1.node('res1', 'Residual Add Layer 1\n[1024×L×8192]', 
               shape='parallelogram', style='filled', fillcolor='orange')
        
        # Connect within stage 1
        c1.edge('layer1_input', 'mha1_q')
        c1.edge('layer1_input', 'mha1_k')
        c1.edge('layer1_input', 'mha1_v')
        c1.edge('mha1_q', 'mha1_attn')
        c1.edge('mha1_k', 'mha1_attn')
        c1.edge('mha1_v', 'mha1_attn')
        c1.edge('mha1_attn', 'gate1')
        c1.edge('mha1_attn', 'res1')
        c1.edge('expert1_agg', 'res1')
    
    # Pipeline communication
    dot.node('pipeline_comm', 'Pipeline Communication\nSend/Recv between stages\n[1024×L×8192]', 
            shape='parallelogram', style='filled', fillcolor='purple')
    
    # Connect pipeline stages
    dot.edge('input', 'layer0_input')
    dot.edge('res0', 'pipeline_comm')
    dot.edge('pipeline_comm', 'layer1_input')
    
    # Output
    dot.node('output', 'Output\n[1024×L×8192]', 
            shape='ellipse', style='filled', fillcolor='lightblue')
    dot.edge('res1', 'output')
    
    # Add TP communication nodes
    with dot.subgraph(name='cluster_tp_comm') as c:
        c.attr(label='Tensor Parallel Communication', style='dashed', bgcolor='white', color='gray')
        c.node('tp_comm_0', 'All-reduce\nStage 0 TP group\nDevices 0-7', 
              shape='parallelogram', style='filled', fillcolor='pink')
        c.node('tp_comm_1', 'All-reduce\nStage 1 TP group\nDevices 8-15', 
              shape='parallelogram', style='filled', fillcolor='pink')
    
    # Connect TP communications
    dot.edge('mha0_attn', 'tp_comm_0', style='dashed')
    dot.edge('tp_comm_0', 'gate0', style='dashed')
    dot.edge('mha1_attn', 'tp_comm_1', style='dashed')
    dot.edge('tp_comm_1', 'gate1', style='dashed')
    
    # Add legend
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='Legend', style='dashed', bgcolor='white', color='gray')
        c.node('legend_input', 'Input/Output', shape='ellipse', style='filled', fillcolor='lightblue')
        c.node('legend_compute', 'Computation', shape='rectangle', style='filled', fillcolor='lightgreen')
        c.node('legend_comm', 'Communication', shape='parallelogram', style='filled', fillcolor='lightyellow')
        c.node('legend_tp', 'Tensor Parallel Comm', shape='parallelogram', style='filled', fillcolor='pink')
        c.node('legend_pp', 'Pipeline Comm', shape='parallelogram', style='filled', fillcolor='purple')
        c.node('legend_routing', 'Expert Routing', style='dashed')
    
    return dot

if __name__ == '__main__':
    dag = create_helix_moe_baseline_dag()
    dag.render()
    print("Generated helix_moe_baseline_dag.svg")