import graphviz
from typing import Dict, List, Tuple

def create_baseline_dag():
    """Create DAG for baseline configuration: 16 GPUs, TP=8, PP=2, 4 experts per GPU"""
    dot = graphviz.Digraph('baseline_moe', comment='MoE Baseline Deployment (16 GPUs, TP=8, PP=2)')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='1.0')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    dot.attr('edge', arrowhead='normal')
    
    # Global input
    dot.node('input', 'Input Tokens\n[1024, seq_len, 8192]\nAll GPUs', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Pipeline Stage 0 (Layers 0-1) on GPUs 0-7
    with dot.subgraph(name='cluster_stage0') as c:
        c.attr(label='Pipeline Stage 0 (Layers 0-1)\nGPUs 0-7', style='dashed')
        
        # Layer 0
        with c.subgraph(name='cluster_layer0') as l0:
            l0.attr(label='Layer 0', style='dotted')
            
            # Attention across 8 GPUs (TP=8)
            l0.node('l0_attn_qkv', 'QKV Linear\n[1024, seq_len, 8192] -> [1024, seq_len, 24576]\nTP=8 across GPUs 0-7', 
                   fillcolor='lightcoral')
            l0.node('l0_attn_split', 'Split Heads\n[1024, seq_len, 24576] -> [1024, 16, seq_len, 1536]\nAll GPUs', 
                   shape='ellipse', fillcolor='yellow')
            l0.node('l0_attn_score', 'Attention Score\n[1024, 16, seq_len, seq_len]\nAll GPUs', 
                   fillcolor='lightcoral')
            l0.node('l0_attn_out', 'Attention Output\n[1024, 16, seq_len, 1536] -> [1024, seq_len, 8192]\nTP=8 across GPUs 0-7', 
                   fillcolor='lightcoral')
            l0.node('l0_attn_residual', 'Add & Norm\n[1024, seq_len, 8192]\nAll GPUs', 
                   shape='parallelogram', fillcolor='lightgreen')
            
            # MoE Layer - 4 experts per GPU
            for gpu in range(8):
                l0.node(f'l0_gate_{gpu}', f'Gate\nGPU {gpu}', 
                       shape='parallelogram', fillcolor='orange')
                for expert in range(4):
                    l0.node(f'l0_expert_{gpu}_{expert}', 
                           f'Expert {expert}\n[1024, seq_len, 8192] -> [1024, seq_len, 8192]\nGPU {gpu}', 
                           fillcolor='lightblue')
                l0.node(f'l0_mlp_out_{gpu}', f'MoE Output\n[1024, seq_len, 8192]\nGPU {gpu}', 
                       shape='parallelogram', fillcolor='lightgreen')
    
    # Pipeline Stage 1 (Layers 2-3) on GPUs 8-15
    with dot.subgraph(name='cluster_stage1') as c:
        c.attr(label='Pipeline Stage 1 (Layers 2-3)\nGPUs 8-15', style='dashed')
        
        # Layer 2
        with c.subgraph(name='cluster_layer2') as l2:
            l2.attr(label='Layer 2', style='dotted')
            
            # Attention across 8 GPUs (TP=8)
            l2.node('l2_attn_qkv', 'QKV Linear\n[1024, seq_len, 8192] -> [1024, seq_len, 24576]\nTP=8 across GPUs 8-15', 
                   fillcolor='lightcoral')
            l2.node('l2_attn_split', 'Split Heads\n[1024, seq_len, 24576] -> [1024, 16, seq_len, 1536]\nAll GPUs', 
                   shape='ellipse', fillcolor='yellow')
            l2.node('l2_attn_score', 'Attention Score\n[1024, 16, seq_len, seq_len]\nAll GPUs', 
                   fillcolor='lightcoral')
            l2.node('l2_attn_out', 'Attention Output\n[1024, 16, seq_len, 1536] -> [1024, seq_len, 8192]\nTP=8 across GPUs 8-15', 
                   fillcolor='lightcoral')
            l2.node('l2_attn_residual', 'Add & Norm\n[1024, seq_len, 8192]\nAll GPUs', 
                   shape='parallelogram', fillcolor='lightgreen')
            
            # MoE Layer - 4 experts per GPU
            for gpu in range(8, 16):
                l2.node(f'l2_gate_{gpu}', f'Gate\nGPU {gpu}', 
                       shape='parallelogram', fillcolor='orange')
                for expert in range(4):
                    l2.node(f'l2_expert_{gpu}_{expert}', 
                           f'Expert {expert}\n[1024, seq_len, 8192] -> [1024, seq_len, 8192]\nGPU {gpu}', 
                           fillcolor='lightblue')
                l2.node(f'l2_mlp_out_{gpu}', f'MoE Output\n[1024, seq_len, 8192]\nGPU {gpu}', 
                       shape='parallelogram', fillcolor='lightgreen')
    
    # Global output
    dot.node('output', 'Output Tokens\n[1024, seq_len, 8192]\nAll GPUs', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Connections
    # Input to Layer 0
    dot.edge('input', 'l0_attn_qkv')
    dot.edge('l0_attn_qkv', 'l0_attn_split')
    dot.edge('l0_attn_split', 'l0_attn_score')
    dot.edge('l0_attn_score', 'l0_attn_out')
    dot.edge('l0_attn_out', 'l0_attn_residual')
    dot.edge('input', 'l0_attn_residual', style='dashed')  # Residual connection
    
    # Layer 0 MoE
    for gpu in range(8):
        dot.edge('l0_attn_residual', f'l0_gate_{gpu}')
        for expert in range(4):
            dot.edge(f'l0_gate_{gpu}', f'l0_expert_{gpu}_{expert}', style='dashed')
            dot.edge('l0_attn_residual', f'l0_expert_{gpu}_{expert}')
            dot.edge(f'l0_expert_{gpu}_{expert}', f'l0_mlp_out_{gpu}')
        dot.edge('l0_attn_residual', f'l0_mlp_out_{gpu}', style='dashed')  # Residual connection
    
    # Pipeline communication
    dot.edge('l0_mlp_out_0', 'l2_attn_qkv', lhead='cluster_stage1')
    
    # Layer 2
    dot.edge('l2_attn_qkv', 'l2_attn_split')
    dot.edge('l2_attn_split', 'l2_attn_score')
    dot.edge('l2_attn_score', 'l2_attn_out')
    dot.edge('l2_attn_out', 'l2_attn_residual')
    dot.edge('l0_mlp_out_0', 'l2_attn_residual', style='dashed')  # Residual connection
    
    # Layer 2 MoE
    for gpu in range(8, 16):
        dot.edge('l2_attn_residual', f'l2_gate_{gpu}')
        for expert in range(4):
            dot.edge(f'l2_gate_{gpu}', f'l2_expert_{gpu}_{expert}', style='dashed')
            dot.edge('l2_attn_residual', f'l2_expert_{gpu}_{expert}')
            dot.edge(f'l2_expert_{gpu}_{expert}', f'l2_mlp_out_{gpu}')
        dot.edge('l2_attn_residual', f'l2_mlp_out_{gpu}', style='dashed')  # Residual connection
    
    # Output
    dot.edge('l2_mlp_out_8', 'output')
    
    return dot

def create_proposed_dag():
    """Create DAG for proposed configuration: 64 GPUs, EP=64, 1 expert per GPU"""
    dot = graphviz.Digraph('proposed_moe', comment='MoE Proposed Deployment (64 GPUs, EP=64)')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.3', ranksep='0.8')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')
    dot.attr('edge', arrowhead='normal')
    
    # Global input
    dot.node('input', 'Input Tokens\n[1024, seq_len, 8192]\nAll GPUs', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Create all 4 layers
    for layer in range(4):
        layer_name = f'layer{layer}'
        
        with dot.subgraph(name=f'cluster_{layer_name}') as c:
            c.attr(label=f'Layer {layer} (EP=64)', style='dashed')
            
            # Attention across all GPUs (replicated)
            c.node(f'{layer_name}_attn_qkv', f'QKV Linear\n[1024, seq_len, 8192] -> [1024, seq_len, 24576]\nAll 64 GPUs', 
                  fillcolor='lightcoral')
            c.node(f'{layer_name}_attn_split', f'Split Heads\n[1024, seq_len, 24576] -> [1024, 16, seq_len, 1536]\nAll GPUs', 
                  shape='ellipse', fillcolor='yellow')
            c.node(f'{layer_name}_attn_score', f'Attention Score\n[1024, 16, seq_len, seq_len]\nAll GPUs', 
                  fillcolor='lightcoral')
            c.node(f'{layer_name}_attn_out', f'Attention Output\n[1024, 16, seq_len, 1536] -> [1024, seq_len, 8192]\nAll 64 GPUs', 
                  fillcolor='lightcoral')
            c.node(f'{layer_name}_attn_residual', f'Add & Norm\n[1024, seq_len, 8192]\nAll GPUs', 
                  shape='parallelogram', fillcolor='lightgreen')
            
            # MoE Layer - 1 expert per GPU (64 experts total)
            c.node(f'{layer_name}_gate', f'Gate\n[1024, seq_len, 8192] -> routing decisions\nAll GPUs', 
                  shape='parallelogram', fillcolor='orange')
            
            # Create experts distributed across 64 GPUs
            for gpu in range(64):
                expert_id = layer * 64 + gpu
                c.node(f'{layer_name}_expert_{gpu}', 
                      f'Expert {expert_id}\n[selected_tokens, 8192] -> [selected_tokens, 8192]\nGPU {gpu}', 
                      fillcolor='lightblue')
            
            c.node(f'{layer_name}_mlp_aggregate', f'MoE Aggregate\n[1024, seq_len, 8192]\nAll GPUs', 
                  shape='parallelogram', fillcolor='lightgreen')
    
    # Global output
    dot.node('output', 'Output Tokens\n[1024, seq_len, 8192]\nAll GPUs', 
             shape='parallelogram', fillcolor='lightgreen')
    
    # Connections for each layer
    prev_output = 'input'
    for layer in range(4):
        layer_name = f'layer{layer}'
        
        # Attention path
        dot.edge(prev_output, f'{layer_name}_attn_qkv')
        dot.edge(f'{layer_name}_attn_qkv', f'{layer_name}_attn_split')
        dot.edge(f'{layer_name}_attn_split', f'{layer_name}_attn_score')
        dot.edge(f'{layer_name}_attn_score', f'{layer_name}_attn_out')
        dot.edge(f'{layer_name}_attn_out', f'{layer_name}_attn_residual')
        dot.edge(prev_output, f'{layer_name}_attn_residual', style='dashed')  # Residual connection
        
        # MoE path
        dot.edge(f'{layer_name}_attn_residual', f'{layer_name}_gate')
        
        # Gate to experts (dashed lines for routing)
        for gpu in range(64):
            dot.edge(f'{layer_name}_gate', f'{layer_name}_expert_{gpu}', style='dashed')
            dot.edge(f'{layer_name}_attn_residual', f'{layer_name}_expert_{gpu}')
            dot.edge(f'{layer_name}_expert_{gpu}', f'{layer_name}_mlp_aggregate')
        
        # Residual connection for MoE
        dot.edge(f'{layer_name}_attn_residual', f'{layer_name}_mlp_aggregate', style='dashed')
        
        prev_output = f'{layer_name}_mlp_aggregate'
    
    # Final output
    dot.edge(prev_output, 'output')
    
    return dot

if __name__ == "__main__":
    # Generate baseline DAG
    baseline_dag = create_baseline_dag()
    baseline_dag.render('/home/wzc/data/file-share/submission/baseline_moe', format='svg', cleanup=True)
    
    # Generate proposed DAG
    proposed_dag = create_proposed_dag()
    proposed_dag.render('/home/wzc/data/file-share/submission/proposed_moe', format='svg', cleanup=True)
    
    print("DAGs generated successfully!")
    print("Baseline DAG: /home/wzc/data/file-share/submission/baseline_moe.svg")
    print("Proposed DAG: /home/wzc/data/file-share/submission/proposed_moe.svg")