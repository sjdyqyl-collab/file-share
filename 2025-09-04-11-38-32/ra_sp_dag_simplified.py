#!/usr/bin/env python3

import graphviz

def create_ra_sp_dag_simplified():
    """
    Create simplified but accurate DAG for Ring Attention with Sequence Parallelism (RA+SP)
    Total 16 GPUs with sequence parallelism (SP=16) and ring attention
    """
    dot = graphviz.Digraph(comment='Ring Attention with Sequence Parallelism DAG')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input node
    dot.node('input', 'Input\n(B=1024, L=16384, d=8192)\nAll GPUs',
             shape='parallelogram', fillcolor='lightgreen')
    
    # Sequence split across 16 GPUs
    dot.node('seq_split', 'Sequence Split\n(B=1024, L=1024, d=8192)\n16 chunks\nAll GPUs',
             shape='ellipse', fillcolor='orange')
    
    # Embedding layer
    dot.node('embed', 'Embedding\n(B=1024, L=1024, d=8192)\nPer GPU\nAll GPUs 0-15',
             shape='rectangle', fillcolor='lightyellow')
    
    # Process 4 layers with Ring Attention
    for layer_id in range(4):
        color = 'lightcoral' if layer_id % 2 == 0 else 'lightblue'
        dot.attr('node', fillcolor=color)
        
        # QKV projection
        dot.node(f'l{layer_id}_qkv', f'Layer{layer_id} QKV Proj\n(B=1024, L=1024, d=8192)\nAll GPUs 0-15',
                 shape='rectangle')
        
        # Ring Attention - simplified representation
        dot.node(f'l{layer_id}_ring_attn', f'Layer{layer_id} Ring Attention\n(B=1024, L=1024, d=8192)\n16 stages\nAll GPUs 0-15',
                 shape='rectangle')
        
        # Ring communication pattern
        dot.node(f'l{layer_id}_ring_comm', f'Layer{layer_id} Ring Communication\nKV Exchange\nRing Topology\nAll GPUs 0-15',
                 shape='ellipse', fillcolor='orange')
        
        # Attention output
        dot.node(f'l{layer_id}_attn_out', f'Layer{layer_id} Attention Out\n(B=1024, L=1024, d=8192)\nAll GPUs 0-15',
                 shape='rectangle')
        
        # Residual connection
        dot.node(f'l{layer_id}_attn_res', f'Layer{layer_id} Attention Residual\n(B=1024, L=1024, d=8192)\nAll GPUs 0-15',
                 shape='diamond', fillcolor='lightgreen')
        
        # MLP layers
        dot.node(f'l{layer_id}_mlp_up', f'Layer{layer_id} MLP Up\n(B=1024, L=1024, d=32768)\nAll GPUs 0-15',
                 shape='rectangle')
        dot.node(f'l{layer_id}_mlp_down', f'Layer{layer_id} MLP Down\n(B=1024, L=1024, d=8192)\nAll GPUs 0-15',
                 shape='rectangle')
        dot.node(f'l{layer_id}_mlp_res', f'Layer{layer_id} MLP Residual\n(B=1024, L=1024, d=8192)\nAll GPUs 0-15',
                 shape='diamond', fillcolor='lightgreen')
    
    # Sequence gather
    dot.node('seq_gather', 'Sequence Gather\n(B=1024, L=16384, d=8192)\n16 chunks → full\nAll GPUs 0-15',
             shape='ellipse', fillcolor='orange')
    
    # LM Head
    dot.node('lm_head', 'LM Head\n(B=1024, L=1024, d=32000)\nPer GPU\nAll GPUs 0-15',
             shape='rectangle', fillcolor='lightyellow')
    
    # Output
    dot.node('output', 'Output Gather\n(B=1024, L=16384, V=32000)\nAll GPUs',
             shape='parallelogram', fillcolor='lightgreen')
    
    # Connect the nodes
    dot.edge('input', 'seq_split')
    dot.edge('seq_split', 'embed')
    
    # Connect each layer
    prev_node = 'embed'
    for layer_id in range(4):
        # QKV projection
        dot.edge(prev_node, f'l{layer_id}_qkv')
        
        # Ring attention
        dot.edge(f'l{layer_id}_qkv', f'l{layer_id}_ring_attn')
        dot.edge(f'l{layer_id}_ring_attn', f'l{layer_id}_ring_comm')
        dot.edge(f'l{layer_id}_ring_comm', f'l{layer_id}_attn_out')
        
        # Residual and MLP
        dot.edge(prev_node, f'l{layer_id}_attn_res')
        dot.edge(f'l{layer_id}_attn_out', f'l{layer_id}_attn_res')
        dot.edge(f'l{layer_id}_attn_res', f'l{layer_id}_mlp_up')
        dot.edge(f'l{layer_id}_mlp_up', f'l{layer_id}_mlp_down')
        dot.edge(f'l{layer_id}_attn_res', f'l{layer_id}_mlp_res')
        dot.edge(f'l{layer_id}_mlp_down', f'l{layer_id}_mlp_res')
        
        prev_node = f'l{layer_id}_mlp_res'
    
    # Final connections
    dot.edge(prev_node, 'seq_gather')
    dot.edge('seq_gather', 'lm_head')
    dot.edge('lm_head', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_ra_sp_dag_simplified()
    dag.render('/home/wzc/data/file-share/2025-09-04-11-38-32/ra_sp_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-04-11-38-32/ra_sp_dag.dot')
    print("RA+SP DAG generated successfully")