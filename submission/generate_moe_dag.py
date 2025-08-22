import graphviz
from graphviz import Digraph

def create_baseline_dag():
    """Create baseline DAG with TP=8, PP=2, 16 GPUs total, 4 experts per GPU"""
    dot = Digraph(comment='MoE Baseline Deployment (TP=8, PP=2, 16 GPUs)')
    dot.attr(rankdir='TB', size='20,30')
    
    # Colors for different GPU groups
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink']
    
    # Input
    dot.node('input', 'Input Tokens\n[batch=1024, seq_len, hidden=8192]', shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Pipeline Stage 1 (Layers 0-1)
    with dot.subgraph(name='cluster_stage1') as c:
        c.attr(label='Pipeline Stage 1\n(Layers 0-1)', style='dashed', color='black')
        
        # 8 GPUs for stage 1
        for gpu_id in range(8):
            gpu_name = f'gpu{gpu_id}_stage1'
            c.node(gpu_name, f'GPU {gpu_id}\n(Layers 0-1)', shape='rectangle', style='filled', fillcolor=colors[gpu_id % len(colors)])
            
            # Each GPU has 4 experts
            for expert_id in range(4):
                expert_name = f'expert{expert_id}_gpu{gpu_id}_stage1'
                c.node(expert_name, f'Expert {expert_id}\n[hidden=8192, ffn=32768]\nTP=8 shard', shape='rectangle', style='filled', fillcolor='white')
                c.edge(gpu_name, expert_name, style='dashed')
    
    # Pipeline Stage 2 (Layers 2-3)
    with dot.subgraph(name='cluster_stage2') as c:
        c.attr(label='Pipeline Stage 2\n(Layers 2-3)', style='dashed', color='black')
        
        # 8 GPUs for stage 2
        for gpu_id in range(8, 16):
            gpu_name = f'gpu{gpu_id}_stage2'
            c.node(gpu_name, f'GPU {gpu_id}\n(Layers 2-3)', shape='rectangle', style='filled', fillcolor=colors[gpu_id % len(colors)])
            
            # Each GPU has 4 experts
            for expert_id in range(4):
                expert_name = f'expert{expert_id}_gpu{gpu_id}_stage2'
                c.node(expert_name, f'Expert {expert_id}\n[hidden=8192, ffn=32768]\nTP=8 shard', shape='rectangle', style='filled', fillcolor='white')
                c.edge(gpu_name, expert_name, style='dashed')
    
    # Add routing nodes
    dot.node('gate0', 'Gating Layer 0\nSelect top-K experts', shape='parallelogram', style='filled', fillcolor='orange')
    dot.node('gate1', 'Gating Layer 1\nSelect top-K experts', shape='parallelogram', style='filled', fillcolor='orange')
    dot.node('gate2', 'Gating Layer 2\nSelect top-K experts', shape='parallelogram', style='filled', fillcolor='orange')
    dot.node('gate3', 'Gating Layer 3\nSelect top-K experts', shape='parallelogram', style='filled', fillcolor='orange')
    
    # Add attention nodes for each layer
    for layer in range(4):
        dot.node(f'mha{layer}', f'Multi-Head Attention Layer {layer}\n[16 heads, 512 dim/head]\nTP=8', shape='rectangle', style='filled', fillcolor='lightcyan')
        dot.node(f'residual{layer}', f'Residual Add Layer {layer}', shape='ellipse', style='filled', fillcolor='lightcoral')
    
    # Connections
    dot.edge('input', 'mha0')
    dot.edge('mha0', 'gate0')
    
    # Connect gates to experts in their stage
    for layer in [0, 1]:
        for gpu_id in range(8):
            for expert_id in range(4):
                dot.edge(f'gate{layer}', f'expert{expert_id}_gpu{gpu_id}_stage1', style='dashed', label=f'Layer {layer}')
    
    for layer in [2, 3]:
        for gpu_id in range(8, 16):
            for expert_id in range(4):
                dot.edge(f'gate{layer}', f'expert{expert_id}_gpu{gpu_id}_stage2', style='dashed', label=f'Layer {layer}')
    
    # Pipeline connections
    dot.edge('gate0', 'residual0')
    dot.edge('residual0', 'mha1')
    dot.edge('mha1', 'gate1')
    dot.edge('gate1', 'residual1')
    dot.edge('residual1', 'mha2')
    dot.edge('mha2', 'gate2')
    dot.edge('gate2', 'residual2')
    dot.edge('residual2', 'mha3')
    dot.edge('mha3', 'gate3')
    dot.edge('gate3', 'residual3')
    
    # Add communication nodes
    dot.node('comm_stage1_to_2', 'Pipeline Communication\nStage 1 → Stage 2', shape='ellipse', style='filled', fillcolor='yellow')
    dot.edge('residual1', 'comm_stage1_to_2')
    dot.edge('comm_stage1_to_2', 'mha2')
    
    # Output
    dot.node('output', 'Output Tokens\n[batch=1024, seq_len, hidden=8192]', shape='ellipse', style='filled', fillcolor='lightgray')
    dot.edge('residual3', 'output')
    
    return dot

def create_proposed_dag():
    """Create proposed DAG with 64 GPUs, 1 expert per GPU"""
    dot = Digraph(comment='MoE Proposed Deployment (64 GPUs, 1 Expert/GPU)')
    dot.attr(rankdir='TB', size='30,40')
    
    # Input
    dot.node('input', 'Input Tokens\n[batch=1024, seq_len, hidden=8192]', shape='ellipse', style='filled', fillcolor='lightgray')
    
    # Create all 64 GPUs with 1 expert each
    experts = []
    for layer in range(4):
        with dot.subgraph(name=f'cluster_layer{layer}') as c:
            c.attr(label=f'Layer {layer} - 16 Experts (16 GPUs)', style='dashed', color='black')
            
            for expert_id in range(16):
                gpu_id = layer * 16 + expert_id
                gpu_name = f'gpu{gpu_id}_layer{layer}'
                expert_name = f'expert{expert_id}_layer{layer}'
                
                c.node(gpu_name, f'GPU {gpu_id}\nLayer {layer}', shape='rectangle', style='filled', fillcolor='lightblue')
                c.node(expert_name, f'Expert {expert_id}\n[hidden=8192, ffn=32768]', shape='rectangle', style='filled', fillcolor='white')
                c.edge(gpu_name, expert_name)
                experts.append((layer, expert_id, gpu_id))
    
    # Add routing and attention for each layer
    for layer in range(4):
        dot.node(f'mha{layer}', f'Multi-Head Attention Layer {layer}\n[16 heads, 512 dim/head]', shape='rectangle', style='filled', fillcolor='lightcyan')
        dot.node(f'gate{layer}', f'Gating Layer {layer}\nSelect top-K experts', shape='parallelogram', style='filled', fillcolor='orange')
        dot.node(f'residual{layer}', f'Residual Add Layer {layer}', shape='ellipse', style='filled', fillcolor='lightcoral')
        
        # Add token aggregation nodes for each layer
        dot.node(f'aggregate{layer}', f'Token Aggregation\nLayer {layer}', shape='ellipse', style='filled', fillcolor='yellow')
    
    # Connections
    dot.edge('input', 'mha0')
    dot.edge('mha0', 'gate0')
    
    # Connect gates to all experts in their layer
    for layer in range(4):
        for expert_id in range(16):
            gpu_id = layer * 16 + expert_id
            dot.edge(f'gate{layer}', f'expert{expert_id}_layer{layer}', style='dashed', label='token routing')
    
    # Connect experts to aggregation
    for layer in range(4):
        for expert_id in range(16):
            dot.edge(f'expert{expert_id}_layer{layer}', f'aggregate{layer}')
        dot.edge(f'aggregate{layer}', f'residual{layer}')
        if layer < 3:
            dot.edge(f'residual{layer}', f'mha{layer+1}')
        else:
            dot.edge('residual3', 'output')
    
    # Add inter-GPU communication nodes
    for layer in range(4):
        for expert_id in range(16):
            for other_expert in range(16):
                if expert_id != other_expert:
                    comm_name = f'comm_layer{layer}_expert{expert_id}_to_{other_expert}'
                    dot.node(comm_name, f'Cross-GPU Comm\nLayer {layer}\nExpert {expert_id}→{other_expert}', shape='ellipse', style='filled', fillcolor='lightyellow')
                    dot.edge(f'expert{expert_id}_layer{layer}', comm_name)
                    dot.edge(comm_name, f'aggregate{layer}')
    
    # Output
    dot.node('output', 'Output Tokens\n[batch=1024, seq_len, hidden=8192]', shape='ellipse', style='filled', fillcolor='lightgray')
    
    return dot

# Generate both DAGs
if __name__ == "__main__":
    # Create baseline DAG
    baseline_dag = create_baseline_dag()
    baseline_dag.render('/home/wzc/data/file-share/submission/moe_baseline_dag', format='svg', cleanup=True)
    
    # Create proposed DAG
    proposed_dag = create_proposed_dag()
    proposed_dag.render('/home/wzc/data/file-share/submission/moe_proposed_dag', format='svg', cleanup=True)
    
    print("DAGs generated successfully!")
    print("Baseline DAG saved to: /home/wzc/data/file-share/submission/moe_baseline_dag.svg")
    print("Proposed DAG saved to: /home/wzc/data/file-share/submission/moe_proposed_dag.svg")