#!/usr/bin/env python3

import graphviz
from graphviz import Digraph

def create_proposed_dag():
    """Create proposed DAG with EP=64, 64 GPUs, 1 expert per GPU"""
    
    dot = Digraph(comment='Proposed Cross-Node Expert Parallelism: EP=64')
    dot.attr(rankdir='TB', size='30,40', fontname='Arial')
    
    # Define node styles
    dot.attr('node', shape='rectangle', style='filled', fontname='Arial')
    dot.attr('edge', fontname='Arial', fontsize='10')
    
    # Colors for different nodes
    colors = {
        'attention': 'lightblue',
        'expert': 'lightgreen',
        'gate': 'lightyellow',
        'router': 'orange',
        'communication': 'yellow',
        'input': 'lightgrey',
        'output': 'lightcoral',
        'aggregation': 'lightpink'
    }
    
    # Global input
    dot.node('input', 'Input\n[1024, 2048, 8192]', fillcolor=colors['input'])
    
    # Define all 4 layers with 16 experts each = 64 experts total
    # Each expert on a separate GPU
    
    # Layer 0 - Experts 0-15 on GPUs 0-15 (Node 0-1)
    with dot.subgraph(name='cluster_layer0') as layer0:
        layer0.attr(label='Layer 0 - Experts 0-15 (Nodes 0-1)', style='dashed', color='blue')
        
        # Attention (replicated across all nodes)
        layer0.node('layer0_attn', 'Multi-Head Attention\n[1024, 8192] -> [1024, 8192]\nReplicated on all GPUs', fillcolor=colors['attention'])
        layer0.node('layer0_attn_residual', 'Residual Add\n[1024, 8192] + [1024, 8192]\nAll GPUs', fillcolor=colors['attention'])
        
        # Gate network
        layer0.node('layer0_gate', 'Gate Network\n[1024, 8192] -> [1024, 16]\nReplicated', fillcolor=colors['gate'])
        
        # Token router
        layer0.node('layer0_router', 'Token Router\n[1024, 8192] -> [tokens, 8192]\nAsync routing to experts', fillcolor=colors['router'], shape='parallelogram')
        
        # 16 experts on 16 different GPUs
        for expert_id in range(16):
            gpu_id = expert_id
            node_id = gpu_id // 8
            layer0.node(f'layer0_expert{expert_id}', 
                       f'Expert {expert_id}\n[tokens, 8192] -> [tokens, 32768] -> [tokens, 8192]\nGPU: {gpu_id} (Node {node_id})', 
                       fillcolor=colors['expert'])
        
        # Expert aggregation
        layer0.node('layer0_aggregate', 'Expert Aggregation\n[tokens, 8192] x k=2\nCross-node gather', fillcolor=colors['aggregation'], shape='parallelogram')
        layer0.node('layer0_moe_residual', 'Residual Add\n[1024, 8192] + [1024, 8192]\nAll GPUs', fillcolor=colors['attention'])
    
    # Communication to Layer 1
    dot.node('comm0_1', 'Layer 0 -> Layer 1\n[1024, 8192]\nCross-node transfer', fillcolor=colors['communication'], shape='ellipse')
    
    # Layer 1 - Experts 16-31 on GPUs 16-31 (Node 2-3)
    with dot.subgraph(name='cluster_layer1') as layer1:
        layer1.attr(label='Layer 1 - Experts 16-31 (Nodes 2-3)', style='dashed', color='green')
        
        layer1.node('layer1_attn', 'Multi-Head Attention\n[1024, 8192] -> [1024, 8192]\nReplicated on all GPUs', fillcolor=colors['attention'])
        layer1.node('layer1_attn_residual', 'Residual Add\n[1024, 8192] + [1024, 8192]\nAll GPUs', fillcolor=colors['attention'])
        
        layer1.node('layer1_gate', 'Gate Network\n[1024, 8192] -> [1024, 16]\nReplicated', fillcolor=colors['gate'])
        layer1.node('layer1_router', 'Token Router\n[1024, 8192] -> [tokens, 8192]\nAsync routing to experts', fillcolor=colors['router'], shape='parallelogram')
        
        for expert_id in range(16, 32):
            gpu_id = expert_id
            node_id = gpu_id // 8
            layer1.node(f'layer1_expert{expert_id}', 
                       f'Expert {expert_id}\n[tokens, 8192] -> [tokens, 32768] -> [tokens, 8192]\nGPU: {gpu_id} (Node {node_id})', 
                       fillcolor=colors['expert'])
        
        layer1.node('layer1_aggregate', 'Expert Aggregation\n[tokens, 8192] x k=2\nCross-node gather', fillcolor=colors['aggregation'], shape='parallelogram')
        layer1.node('layer1_moe_residual', 'Residual Add\n[1024, 8192] + [1024, 8192]\nAll GPUs', fillcolor=colors['attention'])
    
    # Communication to Layer 2
    dot.node('comm1_2', 'Layer 1 -> Layer 2\n[1024, 8192]\nCross-node transfer', fillcolor=colors['communication'], shape='ellipse')
    
    # Layer 2 - Experts 32-47 on GPUs 32-47 (Node 4-5)
    with dot.subgraph(name='cluster_layer2') as layer2:
        layer2.attr(label='Layer 2 - Experts 32-47 (Nodes 4-5)', style='dashed', color='red')
        
        layer2.node('layer2_attn', 'Multi-Head Attention\n[1024, 8192] -> [1024, 8192]\nReplicated on all GPUs', fillcolor=colors['attention'])
        layer2.node('layer2_attn_residual', 'Residual Add\n[1024, 8192] + [1024, 8192]\nAll GPUs', fillcolor=colors['attention'])
        
        layer2.node('layer2_gate', 'Gate Network\n[1024, 8192] -> [1024, 16]\nReplicated', fillcolor=colors['gate'])
        layer2.node('layer2_router', 'Token Router\n[1024, 8192] -> [tokens, 8192]\nAsync routing to experts', fillcolor=colors['router'], shape='parallelogram')
        
        for expert_id in range(32, 48):
            gpu_id = expert_id
            node_id = gpu_id // 8
            layer2.node(f'layer2_expert{expert_id}', 
                       f'Expert {expert_id}\n[tokens, 8192] -> [tokens, 32768] -> [tokens, 8192]\nGPU: {gpu_id} (Node {node_id})', 
                       fillcolor=colors['expert'])
        
        layer2.node('layer2_aggregate', 'Expert Aggregation\n[tokens, 8192] x k=2\nCross-node gather', fillcolor=colors['aggregation'], shape='parallelogram')
        layer2.node('layer2_moe_residual', 'Residual Add\n[1024, 8192] + [1024, 8192]\nAll GPUs', fillcolor=colors['attention'])
    
    # Communication to Layer 3
    dot.node('comm2_3', 'Layer 2 -> Layer 3\n[1024, 8192]\nCross-node transfer', fillcolor=colors['communication'], shape='ellipse')
    
    # Layer 3 - Experts 48-63 on GPUs 48-63 (Node 6-7)
    with dot.subgraph(name='cluster_layer3') as layer3:
        layer3.attr(label='Layer 3 - Experts 48-63 (Nodes 6-7)', style='dashed', color='purple')
        
        layer3.node('layer3_attn', 'Multi-Head Attention\n[1024, 8192] -> [1024, 8192]\nReplicated on all GPUs', fillcolor=colors['attention'])
        layer3.node('layer3_attn_residual', 'Residual Add\n[1024, 8192] + [1024, 8192]\nAll GPUs', fillcolor=colors['attention'])
        
        layer3.node('layer3_gate', 'Gate Network\n[1024, 8192] -> [1024, 16]\nReplicated', fillcolor=colors['gate'])
        layer3.node('layer3_router', 'Token Router\n[1024, 8192] -> [tokens, 8192]\nAsync routing to experts', fillcolor=colors['router'], shape='parallelogram')
        
        for expert_id in range(48, 64):
            gpu_id = expert_id
            node_id = gpu_id // 8
            layer3.node(f'layer3_expert{expert_id}', 
                       f'Expert {expert_id}\n[tokens, 8192] -> [tokens, 32768] -> [tokens, 8192]\nGPU: {gpu_id} (Node {node_id})', 
                       fillcolor=colors['expert'])
        
        layer3.node('layer3_aggregate', 'Expert Aggregation\n[tokens, 8192] x k=2\nCross-node gather', fillcolor=colors['aggregation'], shape='parallelogram')
        layer3.node('layer3_moe_residual', 'Residual Add\n[1024, 8192] + [1024, 8192]\nAll GPUs', fillcolor=colors['attention'])
    
    # Global output
    dot.node('output', 'Output\n[1024, 2048, 8192]', fillcolor=colors['output'])
    
    # Connections
    # Input to Layer 0
    dot.edge('input', 'layer0_attn')
    dot.edge('layer0_attn', 'layer0_attn_residual')
    dot.edge('layer0_attn_residual', 'layer0_gate')
    dot.edge('layer0_gate', 'layer0_router', style='dashed', label='routing decisions')
    
    # Router to experts
    for expert_id in range(16):
        dot.edge('layer0_router', f'layer0_expert{expert_id}', label='tokens')
    
    # Experts to aggregation
    for expert_id in range(16):
        dot.edge(f'layer0_expert{expert_id}', 'layer0_aggregate')
    
    dot.edge('layer0_aggregate', 'layer0_moe_residual')
    dot.edge('layer0_moe_residual', 'comm0_1')
    
    # Layer 0 -> Layer 1
    dot.edge('comm0_1', 'layer1_attn')
    dot.edge('layer1_attn', 'layer1_attn_residual')
    dot.edge('layer1_attn_residual', 'layer1_gate')
    dot.edge('layer1_gate', 'layer1_router', style='dashed', label='routing decisions')
    
    for expert_id in range(16, 32):
        dot.edge('layer1_router', f'layer1_expert{expert_id}', label='tokens')
    
    for expert_id in range(16, 32):
        dot.edge(f'layer1_expert{expert_id}', 'layer1_aggregate')
    
    dot.edge('layer1_aggregate', 'layer1_moe_residual')
    dot.edge('layer1_moe_residual', 'comm1_2')
    
    # Layer 1 -> Layer 2
    dot.edge('comm1_2', 'layer2_attn')
    dot.edge('layer2_attn', 'layer2_attn_residual')
    dot.edge('layer2_attn_residual', 'layer2_gate')
    dot.edge('layer2_gate', 'layer2_router', style='dashed', label='routing decisions')
    
    for expert_id in range(32, 48):
        dot.edge('layer2_router', f'layer2_expert{expert_id}', label='tokens')
    
    for expert_id in range(32, 48):
        dot.edge(f'layer2_expert{expert_id}', 'layer2_aggregate')
    
    dot.edge('layer2_aggregate', 'layer2_moe_residual')
    dot.edge('layer2_moe_residual', 'comm2_3')
    
    # Layer 2 -> Layer 3
    dot.edge('comm2_3', 'layer3_attn')
    dot.edge('layer3_attn', 'layer3_attn_residual')
    dot.edge('layer3_attn_residual', 'layer3_gate')
    dot.edge('layer3_gate', 'layer3_router', style='dashed', label='routing decisions')
    
    for expert_id in range(48, 64):
        dot.edge('layer3_router', f'layer3_expert{expert_id}', label='tokens')
    
    for expert_id in range(48, 64):
        dot.edge(f'layer3_expert{expert_id}', 'layer3_aggregate')
    
    dot.edge('layer3_aggregate', 'layer3_moe_residual')
    dot.edge('layer3_moe_residual', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_proposed_dag()
    dag.render('/home/wzc/data/file-share/2025-09-05-09-22-08/proposed_moe_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/2025-09-05-09-22-08/proposed_moe_dag.dot')
    print("Proposed DAG generated successfully")