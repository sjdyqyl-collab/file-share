#!/usr/bin/env python3

import graphviz

def create_ra_sp_dag():
    """
    Create DAG for Ring Attention with Sequence Parallelism (RA+SP)
    Total 16 GPUs with sequence parallelism (SP=16) and ring attention
    """
    dot = graphviz.Digraph(comment='Ring Attention with Sequence Parallelism DAG')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
    
    # Define node styles
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
    
    # Input node - split across 16 GPUs by sequence dimension
    dot.node('input', 'Input\n(B=1024, L=16384, d=8192)\nSequence Split 16×\nAll GPUs',
             shape='parallelogram', fillcolor='lightgreen')
    
    # Sequence split operation
    dot.node('seq_split', 'Sequence Split\n(B=1024, L=1024, d=8192)\n16 chunks\nEach GPU',
             shape='ellipse', fillcolor='orange')
    
    # Embedding layer - each GPU handles its sequence chunk
    dot.node('embed', 'Embedding\n(B=1024, L=1024, d=8192)\nPer GPU\nAll GPUs 0-15',
             shape='rectangle', fillcolor='lightyellow')
    
    # Process 4 layers with Ring Attention + Sequence Parallelism
    for layer_id in range(4):
        color = 'lightcoral' if layer_id % 2 == 0 else 'lightblue'
        dot.attr('node', fillcolor=color)
        
        # Ring Attention QKV projection
        dot.node(f'l{layer_id}_qkv', f'Layer{layer_id} QKV Proj\n(B=1024, L=1024, d=8192)\nRing SP\nAll GPUs 0-15',
                 shape='rectangle')
        
        # Ring Attention stages - 16 stages for 16 GPUs
        for stage in range(16):
            dot.node(f'l{layer_id}_ring_{stage}', f'Layer{layer_id} Ring Stage{stage}\n(B=1024, L=1024, d=512×16)\nGPU {stage}',
                     shape='rectangle')
            
            if stage > 0:
                # Ring communication between stages
                prev_gpu = (stage - 1) % 16
                dot.node(f'l{layer_id}_comm_{prev_gpu}_{stage}', f'Ring Comm\nKV Transfer\nGPU {prev_gpu} → GPU {stage}',
                         shape='ellipse', fillcolor='orange')
        
        # Ring Attention aggregation
        dot.node(f'l{layer_id}_ring_agg', f'Layer{layer_id} Ring Agg\n(B=1024, L=1024, d=8192)\nAll GPUs 0-15',
                 shape='ellipse', fillcolor='orange')
        
        # Attention output projection
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
    
    # Sequence gather operation
    dot.node('seq_gather', 'Sequence Gather\n(B=1024, L=16384, d=8192)\n16 chunks → full\nAll GPUs 0-15',
             shape='ellipse', fillcolor='orange')
    
    # LM Head - sequence parallel across all GPUs
    dot.node('lm_head', 'LM Head\n(B=1024, L=1024, d=32000)\nPer GPU\nAll GPUs 0-15',
             shape='rectangle', fillcolor='lightyellow')
    
    # Output gather
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
        
        # Ring attention stages
        dot.edge(f'l{layer_id}_qkv', f'l{layer_id}_ring_0')
        
        for stage in range(16):
            if stage < 15:
                # Ring communication
                dot.edge(f'l{layer_id}_ring_{stage}', f'l{layer_id}_comm_{stage}_{(stage+1)%16}')
                dot.edge(f'l{layer_id}_comm_{stage}_{(stage+1)%16}', f'l{layer_id}_ring_{(stage+1)%16}')
            else:
                # Final stage connects to aggregation
                dot.edge(f'l{layer_id}_ring_{stage}', f'l{layer_id}_ring_agg')
        
        # Attention output and residual
        dot.edge(f'l{layer_id}_ring_agg', f'l{layer_id}_attn_out')
        dot.edge(prev_node, f'l{layer_id}_attn_res')
        dot.edge(f'l{layer_id}_attn_out', f'l{layer_id}_attn_res')
        
        # MLP
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
    dag = create_ra_sp_dag()
    dag.render('/home/wzc/data/file-share/2025-09-04-11-38-32/ra_sp_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-04-11-38-32/ra_sp_dag.dot')
    print("RA+SP DAG generated successfully")