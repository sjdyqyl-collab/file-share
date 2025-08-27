#!/usr/bin/env python3

import graphviz

def create_helix_baseline_dag():
    dot = graphviz.Digraph('Helix_Baseline_TP_PP', 
                          filename='helix_baseline_dag.svg',
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
        
        # Attention computation across TP group
        c0.node('mha0_attn', 'MHA Layer 0 Attention\nAll-reduce across 8 GPUs\n[1024×L×8192]', 
               shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # FFN for Layer 0
        c0.node('ffn0_linear1', 'FFN Layer 0 Linear1\n[1024×L×32768] across 8 GPUs', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c0.node('ffn0_act', 'FFN Layer 0 Activation\n[1024×L×32768]', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c0.node('ffn0_linear2', 'FFN Layer 0 Linear2\n[1024×L×8192] across 8 GPUs', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        
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
        c0.edge('mha0_attn', 'ffn0_linear1')
        c0.edge('ffn0_linear1', 'ffn0_act')
        c0.edge('ffn0_act', 'ffn0_linear2')
        c0.edge('mha0_attn', 'res0')
        c0.edge('ffn0_linear2', 'res0')
    
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
        
        # Attention computation across TP group
        c1.node('mha1_attn', 'MHA Layer 1 Attention\nAll-reduce across 8 GPUs\n[1024×L×8192]', 
               shape='rectangle', style='filled', fillcolor='lightcoral')
        
        # FFN for Layer 1
        c1.node('ffn1_linear1', 'FFN Layer 1 Linear1\n[1024×L×32768] across 8 GPUs', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c1.node('ffn1_act', 'FFN Layer 1 Activation\n[1024×L×32768]', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        c1.node('ffn1_linear2', 'FFN Layer 1 Linear2\n[1024×L×8192] across 8 GPUs', 
               shape='rectangle', style='filled', fillcolor='lightgreen')
        
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
        c1.edge('mha1_attn', 'ffn1_linear1')
        c1.edge('ffn1_linear1', 'ffn1_act')
        c1.edge('ffn1_act', 'ffn1_linear2')
        c1.edge('mha1_attn', 'res1')
        c1.edge('ffn1_linear2', 'res1')
    
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
    dot.edge('tp_comm_0', 'ffn0_linear1', style='dashed')
    dot.edge('mha1_attn', 'tp_comm_1', style='dashed')
    dot.edge('tp_comm_1', 'ffn1_linear1', style='dashed')
    
    # Add legend
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='Legend', style='dashed', bgcolor='white', color='gray')
        c.node('legend_input', 'Input/Output', shape='ellipse', style='filled', fillcolor='lightblue')
        c.node('legend_compute', 'Computation', shape='rectangle', style='filled', fillcolor='lightgreen')
        c.node('legend_comm', 'Communication', shape='parallelogram', style='filled', fillcolor='lightyellow')
        c.node('legend_tp', 'Tensor Parallel Comm', shape='parallelogram', style='filled', fillcolor='pink')
        c.node('legend_pp', 'Pipeline Comm', shape='parallelogram', style='filled', fillcolor='purple')
    
    return dot

if __name__ == '__main__':
    dag = create_helix_baseline_dag()
    dag.render()
    print("Generated helix_baseline_dag.svg")