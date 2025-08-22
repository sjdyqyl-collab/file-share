import graphviz

def create_helix_moe_dag():
    dot = graphviz.Digraph('Helix_MoE_Model_DAG', format='svg')
    dot.attr(rankdir='TB', size='35,25')
    
    # Set global attributes
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')
    
    # Color scheme for different operations
    colors = {
        'input': 'lightblue',
        'computation': 'lightgreen',
        'communication': 'lightyellow',
        'aggregation': 'lightcoral',
        'routing': 'lightsteelblue',
        'expert': 'lightseagreen',
        'output': 'lightpink'
    }
    
    # Input layer
    dot.node('input', 'Input\nX: [B=1024, L, D=8192]', shape='ellipse', style='filled', fillcolor=colors['input'])
    
    # Layer 1 (MoE)
    with dot.subgraph(name='cluster_layer1') as c:
        c.attr(label='Layer 1 (MoE)', style='rounded')
        
        # LayerNorm 1
        c.node('ln1', 'LayerNorm\nX: [B=1024, L, 8192]\n→ [B=1024, L, 8192]\nAll GPUs', shape='rectangle', style='filled', fillcolor=colors['computation'])
        
        # MHA with 16 partitions (same as dense)
        for i in range(4):  # head groups
            for j in range(4):  # dimension slices
                gpu_id = i * 4 + j
                
                # Q, K, V projections
                c.node(f'q_proj_{i}_{j}', f'Q Projection\n[B=1024, L, 512] → [B=1024, L, 512]\nGPU {gpu_id}', shape='rectangle', style='filled', fillcolor=colors['computation'])
                c.node(f'k_proj_{i}_{j}', f'K Projection\n[B=1024, L, 512] → [B=1024, L, 512]\nGPU {gpu_id}', shape='rectangle', style='filled', fillcolor=colors['computation'])
                c.node(f'v_proj_{i}_{j}', f'V Projection\n[B=1024, L, 512] → [B=1024, L, 512]\nGPU {gpu_id}', shape='rectangle', style='filled', fillcolor=colors['computation'])
                c.node(f'attn_{i}_{j}', f'Attention\n[B=1024, L, 512] → [B=1024, L, 512]\nGPU {gpu_id}', shape='rectangle', style='filled', fillcolor=colors['computation'])
        
        # MHA aggregation
        for i in range(4):
            c.node(f'agg_dim_{i}', f'Concat Dimensions\n4×512 → 2048\nHead Group {i}\nAll GPUs', shape='parallelogram', style='filled', fillcolor=colors['aggregation'])
        c.node('agg_heads', 'Concat Heads\n4×2048 → 8192\nAll GPUs', shape='parallelogram', style='filled', fillcolor=colors['aggregation'])
        c.node('out_proj1', 'Output Projection\n[B=1024, L, 8192] → [B=1024, L, 8192]\nAll GPUs', shape='rectangle', style='filled', fillcolor=colors['computation'])
        c.node('res1', 'Residual Add\n[B=1024, L, 8192] + [B=1024, L, 8192]\nAll GPUs', shape='ellipse', style='filled', fillcolor=colors['computation'])
        
        # LayerNorm 2
        c.node('ln2', 'LayerNorm\n[B=1024, L, 8192] → [B=1024, L, 8192]\nAll GPUs', shape='rectangle', style='filled', fillcolor=colors['computation'])
        
        # Gate network for expert selection
        c.node('gate', 'Gate Network\n[B=1024, L, 8192] → [B=1024, L, 8]\nTop-2 Expert Selection\nAll GPUs', shape='parallelogram', style='filled', fillcolor=colors['routing'])
        
        # Expert routing (dashed lines for selection)
        for expert_id in range(8):
            c.node(f'expert_{expert_id}', f'Expert {expert_id}\nMLP Block\n[B=1024, L, 8192] → [B=1024, L, 8192]\nGPU {expert_id*2}-{expert_id*2+1}', shape='rectangle', style='filled', fillcolor=colors['expert'])
            
            # Dashed lines for gate selection
            c.edge('gate', f'expert_{expert_id}', style='dashed', label=f'Top-2 routing')
        
        # Expert aggregation
        c.node('expert_agg', 'Weighted Sum\n8×[B=1024, L, 8192] → [B=1024, L, 8192]\nAll GPUs', shape='parallelogram', style='filled', fillcolor=colors['aggregation'])
        
        # Final residual
        c.node('res2', 'Residual Add\n[B=1024, L, 8192] + [B=1024, L, 8192]\nAll GPUs', shape='ellipse', style='filled', fillcolor=colors['computation'])
    
    # Connect Layer 1
    dot.edge('input', 'ln1')
    # MHA connections
    for i in range(4):
        for j in range(4):
            dot.edge('ln1', f'q_proj_{i}_{j}')
            dot.edge('ln1', f'k_proj_{i}_{j}')
            dot.edge('ln1', f'v_proj_{i}_{j}')
            dot.edge(f'q_proj_{i}_{j}', f'attn_{i}_{j}')
            dot.edge(f'k_proj_{i}_{j}', f'attn_{i}_{j}')
            dot.edge(f'v_proj_{i}_{j}', f'attn_{i}_{j}')
            dot.edge(f'attn_{i}_{j}', f'agg_dim_{i}')
    
    for i in range(4):
        dot.edge(f'agg_dim_{i}', 'agg_heads')
    
    dot.edge('agg_heads', 'out_proj1')
    dot.edge('out_proj1', 'res1')
    dot.edge('input', 'res1')  # Residual
    dot.edge('res1', 'ln2')
    dot.edge('ln2', 'gate')
    
    # Expert connections
    for expert_id in range(8):
        dot.edge('ln2', f'expert_{expert_id}')
        dot.edge(f'expert_{expert_id}', 'expert_agg')
    
    dot.edge('expert_agg', 'res2')
    dot.edge('res1', 'res2')  # Residual
    
    # Add remaining MoE layers (2,3,4)
    for layer in [2, 3, 4]:
        with dot.subgraph(name=f'cluster_layer{layer}') as c:
            c.attr(label=f'Layer {layer} (MoE)', style='rounded')
            
            # MHA section (same as layer 1)
            c.node(f'ln{layer}_1', f'LayerNorm {layer}.1\n[B=1024, L, 8192] → [B=1024, L, 8192]\nAll GPUs', shape='rectangle', style='filled', fillcolor=colors['computation'])
            
            for i in range(4):
                for j in range(4):
                    gpu_id = i * 4 + j
                    c.node(f'q_proj_{layer}_{i}_{j}', f'Q Projection\n[B=1024, L, 512] → [B=1024, L, 512]\nGPU {gpu_id}', shape='rectangle', style='filled', fillcolor=colors['computation'])
                    c.node(f'k_proj_{layer}_{i}_{j}', f'K Projection\n[B=1024, L, 512] → [B=1024, L, 512]\nGPU {gpu_id}', shape='rectangle', style='filled', fillcolor=colors['computation'])
                    c.node(f'v_proj_{layer}_{i}_{j}', f'V Projection\n[B=1024, L, 512] → [B=1024, L, 512]\nGPU {gpu_id}', shape='rectangle', style='filled', fillcolor=colors['computation'])
                    c.node(f'attn_{layer}_{i}_{j}', f'Attention\n[B=1024, L, 512] → [B=1024, L, 512]\nGPU {gpu_id}', shape='rectangle', style='filled', fillcolor=colors['computation'])
            
            for i in range(4):
                c.node(f'agg_dim_{layer}_{i}', f'Concat Dimensions\n4×512 → 2048\nHead Group {i}\nAll GPUs', shape='parallelogram', style='filled', fillcolor=colors['aggregation'])
            c.node(f'agg_heads_{layer}', f'Concat Heads\n4×2048 → 8192\nAll GPUs', shape='parallelogram', style='filled', fillcolor=colors['aggregation'])
            c.node(f'out_proj{layer}', f'Output Projection\n[B=1024, L, 8192] → [B=1024, L, 8192]\nAll GPUs', shape='rectangle', style='filled', fillcolor=colors['computation'])
            c.node(f'res{layer}_1', f'Residual Add\n[B=1024, L, 8192] + [B=1024, L, 8192]\nAll GPUs', shape='ellipse', style='filled', fillcolor=colors['computation'])
            
            # MoE section
            c.node(f'ln{layer}_2', f'LayerNorm {layer}.2\n[B=1024, L, 8192] → [B=1024, L, 8192]\nAll GPUs', shape='rectangle', style='filled', fillcolor=colors['computation'])
            c.node(f'gate{layer}', f'Gate Network {layer}\n[B=1024, L, 8192] → [B=1024, L, 8]\nTop-2 Expert Selection\nAll GPUs', shape='parallelogram', style='filled', fillcolor=colors['routing'])
            
            for expert_id in range(8):
                c.node(f'expert_{layer}_{expert_id}', f'Expert {layer}.{expert_id}\nMLP Block\n[B=1024, L, 8192] → [B=1024, L, 8192]\nGPU {expert_id*2}-{expert_id*2+1}', shape='rectangle', style='filled', fillcolor=colors['expert'])
                c.edge(f'gate{layer}', f'expert_{layer}_{expert_id}', style='dashed')
            
            c.node(f'expert_agg_{layer}', f'Weighted Sum\n8×[B=1024, L, 8192] → [B=1024, L, 8192]\nAll GPUs', shape='parallelogram', style='filled', fillcolor=colors['aggregation'])
            c.node(f'res{layer}_2', f'Residual Add\n[B=1024, L, 8192] + [B=1024, L, 8192]\nAll GPUs', shape='ellipse', style='filled', fillcolor=colors['computation'])
    
    # Connect layers
    prev_layer = 1
    for layer in [2, 3, 4]:
        dot.edge(f'res{prev_layer}_2', f'ln{layer}_1')
        
        # MHA connections
        for i in range(4):
            for j in range(4):
                dot.edge(f'ln{layer}_1', f'q_proj_{layer}_{i}_{j}')
                dot.edge(f'ln{layer}_1', f'k_proj_{layer}_{i}_{j}')
                dot.edge(f'ln{layer}_1', f'v_proj_{layer}_{i}_{j}')
                dot.edge(f'q_proj_{layer}_{i}_{j}', f'attn_{layer}_{i}_{j}')
                dot.edge(f'k_proj_{layer}_{i}_{j}', f'attn_{layer}_{i}_{j}')
                dot.edge(f'v_proj_{layer}_{i}_{j}', f'attn_{layer}_{i}_{j}')
                dot.edge(f'attn_{layer}_{i}_{j}', f'agg_dim_{layer}_{i}')
        
        for i in range(4):
            dot.edge(f'agg_dim_{layer}_{i}', f'agg_heads_{layer}')
        
        dot.edge(f'agg_heads_{layer}', f'out_proj{layer}')
        dot.edge(f'out_proj{layer}', f'res{layer}_1')
        dot.edge(f'res{prev_layer}_2', f'res{layer}_1')  # Residual
        dot.edge(f'res{layer}_1', f'ln{layer}_2')
        dot.edge(f'ln{layer}_2', f'gate{layer}')
        
        # Expert connections
        for expert_id in range(8):
            dot.edge(f'ln{layer}_2', f'expert_{layer}_{expert_id}')
            dot.edge(f'expert_{layer}_{expert_id}', f'expert_agg_{layer}')
        
        dot.edge(f'expert_agg_{layer}', f'res{layer}_2')
        dot.edge(f'res{layer}_1', f'res{layer}_2')  # Residual
        
        prev_layer = layer
    
    # Output
    dot.node('output', 'Output\n[B=1024, L, 8192]', shape='ellipse', style='filled', fillcolor=colors['output'])
    dot.edge('res4_2', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_helix_moe_dag()
    dag.render('/home/wzc/data/file-share/submission/helix_moe_model_dag', format='svg', cleanup=True)
    print("MoE model DAG generated successfully")