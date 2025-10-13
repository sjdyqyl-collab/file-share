#!/usr/bin/env python3

import graphviz

def create_ma_separation_dag():
    dot = graphviz.Digraph('MA_Separation_Architecture', 
                          comment='MA Separation: Novel Parallel Strategy for MoE-Attention Co-execution',
                          format='svg')
    
    # Set graph attributes
    dot.attr(rankdir='TB', size='30,20')
    dot.attr('node', shape='rectangle', style='filled', fontname='Arial')
    
    # Color scheme
    attention_color = 'lightblue'
    moe_color = 'lightgreen'
    communication_color = 'lightyellow'
    routing_color = 'lightcoral'
    aggregation_color = 'lightpink'
    
    # Input layer
    dot.node('input', 'Model Input\nInput: [batch_size=1024, seq_len=2048, hidden_dim=4096]\nGPU: all GPUs',
             shape='ellipse', fillcolor='lightgray')
    
    # Layer 1 - Attention Phase
    with dot.subgraph(name='cluster_layer1_attention') as c:
        c.attr(label='Layer 1 - Attention Phase (GPUs 0-7)', style='dashed', color='black')
        
        # Input replication
        c.node('l1_input_replicate', 'Replicate Input\nInput: [1024, 2048, 4096]\nOutput: [1024, 2048, 4096]×8\nGPU: 0-7',
               shape='parallelogram', fillcolor=communication_color)
        
        # QKV projections across 8 GPUs
        for gpu in range(8):
            c.node(f'l1_qkv_gpu{gpu}', f'QKV Projection\nGPU {gpu}\nInput: [1024, 2048, 4096]\nOutput: [1024, 2048, 4, 64]×3\nGPU: {gpu}',
                   fillcolor=attention_color)
        
        # All-reduce for K,V
        c.node('l1_kv_allreduce', 'All-Reduce K,V\nInput: [1024, 2048, 4, 64]×8\nOutput: [1024, 2048, 32, 64]\nGPU: 0-7',
               shape='parallelogram', fillcolor=communication_color)
        
        # Attention computation across 8 GPUs
        for gpu in range(8):
            c.node(f'l1_attn_gpu{gpu}', f'Attention\nGPU {gpu}\nInput: Q[1024,2048,4,64], K[1024,2048,32,64], V[1024,2048,32,64]\nOutput: [1024,2048,4,64]\nGPU: {gpu}',
                   fillcolor=attention_color)
        
        # Output aggregation
        c.node('l1_attn_aggregate', 'All-Reduce Concat\nInput: [1024,2048,4,64]×8\nOutput: [1024,2048,32,64]\nGPU: 0-7',
               shape='parallelogram', fillcolor=aggregation_color)
    
    # Layer 1 - MoE Phase
    with dot.subgraph(name='cluster_layer1_moe') as c:
        c.attr(label='Layer 1 - MoE Phase (GPUs 8-15)', style='dashed', color='black')
        
        # Broadcast attention output
        c.node('l1_moe_broadcast', 'Broadcast Attention Output\nInput: [1024,2048,32,64]\nOutput: [1024,2048,4096]×8\nGPU: 8-15',
               shape='parallelogram', fillcolor=communication_color)
        
        # Gate computation
        c.node('l1_gate', 'Gate Computation\nInput: [1024×2048, 4096]\nOutput: [1024×2048, 16]\nGPU: 8-15',
               shape='diamond', fillcolor=routing_color, style='dashed')
        
        # Expert distribution across 8 GPUs
        for gpu in range(8):
            expert_start = gpu * 2
            expert_end = gpu * 2 + 1
            c.node(f'l1_expert_gpu{gpu}', f'Experts {expert_start},{expert_end}\nGPU {gpu+8}\nInput: [tokens, 4096]\nOutput: [tokens, 4096]\nGPU: {gpu+8}',
                   fillcolor=moe_color)
        
        # Expert aggregation
        c.node('l1_moe_aggregate', 'Expert Output Aggregation\nInput: [tokens, 4096]×16\nOutput: [1024,2048,4096]\nGPU: 8-15',
               shape='parallelogram', fillcolor=aggregation_color)
    
    # Residual connection
    dot.node('l1_residual', 'Residual Add\nInput1: [1024,2048,4096] (input)\nInput2: [1024,2048,4096] (moe output)\nOutput: [1024,2048,4096]\nGPU: all GPUs',
             fillcolor='lightgray')
    
    # Layer 2 - Attention Phase (similar to layer 1)
    with dot.subgraph(name='cluster_layer2_attention') as c:
        c.attr(label='Layer 2 - Attention Phase (GPUs 0-7)', style='dashed', color='black')
        
        c.node('l2_input_replicate', 'Replicate Input\nInput: [1024, 2048, 4096]\nOutput: [1024, 2048, 4096]×8\nGPU: 0-7',
               shape='parallelogram', fillcolor=communication_color)
        
        for gpu in range(8):
            c.node(f'l2_qkv_gpu{gpu}', f'QKV Projection\nGPU {gpu}\nInput: [1024, 2048, 4096]\nOutput: [1024, 2048, 4, 64]×3\nGPU: {gpu}',
                   fillcolor=attention_color)
        
        c.node('l2_kv_allreduce', 'All-Reduce K,V\nInput: [1024, 2048, 4, 64]×8\nOutput: [1024, 2048, 32, 64]\nGPU: 0-7',
               shape='parallelogram', fillcolor=communication_color)
        
        for gpu in range(8):
            c.node(f'l2_attn_gpu{gpu}', f'Attention\nGPU {gpu}\nInput: Q[1024,2048,4,64], K[1024,2048,32,64], V[1024,2048,32,64]\nOutput: [1024,2048,4,64]\nGPU: {gpu}',
                   fillcolor=attention_color)
        
        c.node('l2_attn_aggregate', 'All-Reduce Concat\nInput: [1024,2048,4,64]×8\nOutput: [1024,2048,32,64]\nGPU: 0-7',
               shape='parallelogram', fillcolor=aggregation_color)
    
    # Layer 2 - MoE Phase
    with dot.subgraph(name='cluster_layer2_moe') as c:
        c.attr(label='Layer 2 - MoE Phase (GPUs 8-15)', style='dashed', color='black')
        
        c.node('l2_moe_broadcast', 'Broadcast Attention Output\nInput: [1024,2048,32,64]\nOutput: [1024,2048,4096]×8\nGPU: 8-15',
               shape='parallelogram', fillcolor=communication_color)
        
        c.node('l2_gate', 'Gate Computation\nInput: [1024×2048, 4096]\nOutput: [1024×2048, 16]\nGPU: 8-15',
               shape='diamond', fillcolor=routing_color, style='dashed')
        
        for gpu in range(8):
            expert_start = gpu * 2
            expert_end = gpu * 2 + 1
            c.node(f'l2_expert_gpu{gpu}', f'Experts {expert_start},{expert_end}\nGPU {gpu+8}\nInput: [tokens, 4096]\nOutput: [tokens, 4096]\nGPU: {gpu+8}',
                   fillcolor=moe_color)
        
        c.node('l2_moe_aggregate', 'Expert Output Aggregation\nInput: [tokens, 4096]×16\nOutput: [1024,2048,4096]\nGPU: 8-15',
               shape='parallelogram', fillcolor=aggregation_color)
    
    dot.node('l2_residual', 'Residual Add\nInput1: [1024,2048,4096] (input)\nInput2: [1024,2048,4096] (moe output)\nOutput: [1024,2048,4096]\nGPU: all GPUs',
             fillcolor='lightgray')
    
    # Layer 3 - Attention Phase
    with dot.subgraph(name='cluster_layer3_attention') as c:
        c.attr(label='Layer 3 - Attention Phase (GPUs 0-7)', style='dashed', color='black')
        
        c.node('l3_input_replicate', 'Replicate Input\nInput: [1024, 2048, 4096]\nOutput: [1024, 2048, 4096]×8\nGPU: 0-7',
               shape='parallelogram', fillcolor=communication_color)
        
        for gpu in range(8):
            c.node(f'l3_qkv_gpu{gpu}', f'QKV Projection\nGPU {gpu}\nInput: [1024, 2048, 4096]\nOutput: [1024, 2048, 4, 64]×3\nGPU: {gpu}',
                   fillcolor=attention_color)
        
        c.node('l3_kv_allreduce', 'All-Reduce K,V\nInput: [1024, 2048, 4, 64]×8\nOutput: [1024, 2048, 32, 64]\nGPU: 0-7',
               shape='parallelogram', fillcolor=communication_color)
        
        for gpu in range(8):
            c.node(f'l3_attn_gpu{gpu}', f'Attention\nGPU {gpu}\nInput: Q[1024,2048,4,64], K[1024,2048,32,64], V[1024,2048,32,64]\nOutput: [1024,2048,4,64]\nGPU: {gpu}',
                   fillcolor=attention_color)
        
        c.node('l3_attn_aggregate', 'All-Reduce Concat\nInput: [1024,2048,4,64]×8\nOutput: [1024,2048,32,64]\nGPU: 0-7',
               shape='parallelogram', fillcolor=aggregation_color)
    
    # Layer 3 - MoE Phase
    with dot.subgraph(name='cluster_layer3_moe') as c:
        c.attr(label='Layer 3 - MoE Phase (GPUs 8-15)', style='dashed', color='black')
        
        c.node('l3_moe_broadcast', 'Broadcast Attention Output\nInput: [1024,2048,32,64]\nOutput: [1024,2048,4096]×8\nGPU: 8-15',
               shape='parallelogram', fillcolor=communication_color)
        
        c.node('l3_gate', 'Gate Computation\nInput: [1024×2048, 4096]\nOutput: [1024×2048, 16]\nGPU: 8-15',
               shape='diamond', fillcolor=routing_color, style='dashed')
        
        for gpu in range(8):
            expert_start = gpu * 2
            expert_end = gpu * 2 + 1
            c.node(f'l3_expert_gpu{gpu}', f'Experts {expert_start},{expert_end}\nGPU {gpu+8}\nInput: [tokens, 4096]\nOutput: [tokens, 4096]\nGPU: {gpu+8}',
                   fillcolor=moe_color)
        
        c.node('l3_moe_aggregate', 'Expert Output Aggregation\nInput: [tokens, 4096]×16\nOutput: [1024,2048,4096]\nGPU: 8-15',
               shape='parallelogram', fillcolor=aggregation_color)
    
    dot.node('l3_residual', 'Residual Add\nInput1: [1024,2048,4096] (input)\nInput2: [1024,2048,4096] (moe output)\nOutput: [1024,2048,4096]\nGPU: all GPUs',
             fillcolor='lightgray')
    
    # Layer 4 - Attention Phase
    with dot.subgraph(name='cluster_layer4_attention') as c:
        c.attr(label='Layer 4 - Attention Phase (GPUs 0-7)', style='dashed', color='black')
        
        c.node('l4_input_replicate', 'Replicate Input\nInput: [1024, 2048, 4096]\nOutput: [1024, 2048, 4096]×8\nGPU: 0-7',
               shape='parallelogram', fillcolor=communication_color)
        
        for gpu in range(8):
            c.node(f'l4_qkv_gpu{gpu}', f'QKV Projection\nGPU {gpu}\nInput: [1024, 2048, 4096]\nOutput: [1024, 2048, 4, 64]×3\nGPU: {gpu}',
                   fillcolor=attention_color)
        
        c.node('l4_kv_allreduce', 'All-Reduce K,V\nInput: [1024, 2048, 4, 64]×8\nOutput: [1024, 2048, 32, 64]\nGPU: 0-7',
               shape='parallelogram', fillcolor=communication_color)
        
        for gpu in range(8):
            c.node(f'l4_attn_gpu{gpu}', f'Attention\nGPU {gpu}\nInput: Q[1024,2048,4,64], K[1024,2048,32,64], V[1024,2048,32,64]\nOutput: [1024,2048,4,64]\nGPU: {gpu}',
                   fillcolor=attention_color)
        
        c.node('l4_attn_aggregate', 'All-Reduce Concat\nInput: [1024,2048,4,64]×8\nOutput: [1024,2048,32,64]\nGPU: 0-7',
               shape='parallelogram', fillcolor=aggregation_color)
    
    # Layer 4 - MoE Phase
    with dot.subgraph(name='cluster_layer4_moe') as c:
        c.attr(label='Layer 4 - MoE Phase (GPUs 8-15)', style='dashed', color='black')
        
        c.node('l4_moe_broadcast', 'Broadcast Attention Output\nInput: [1024,2048,32,64]\nOutput: [1024,2048,4096]×8\nGPU: 8-15',
               shape='parallelogram', fillcolor=communication_color)
        
        c.node('l4_gate', 'Gate Computation\nInput: [1024×2048, 4096]\nOutput: [1024×2048, 16]\nGPU: 8-15',
               shape='diamond', fillcolor=routing_color, style='dashed')
        
        for gpu in range(8):
            expert_start = gpu * 2
            expert_end = gpu * 2 + 1
            c.node(f'l4_expert_gpu{gpu}', f'Experts {expert_start},{expert_end}\nGPU {gpu+8}\nInput: [tokens, 4096]\nOutput: [tokens, 4096]\nGPU: {gpu+8}',
                   fillcolor=moe_color)
        
        c.node('l4_moe_aggregate', 'Expert Output Aggregation\nInput: [tokens, 4096]×16\nOutput: [1024,2048,4096]\nGPU: 8-15',
               shape='parallelogram', fillcolor=aggregation_color)
    
    dot.node('l4_residual', 'Residual Add\nInput1: [1024,2048,4096] (input)\nInput2: [1024,2048,4096] (moe output)\nOutput: [1024,2048,4096]\nGPU: all GPUs',
             fillcolor='lightgray')
    
    # Output layer
    dot.node('output', 'Model Output\nInput: [1024, 2048, 4096]\nOutput: [1024, 2048, 50265]\nGPU: all GPUs',
             shape='ellipse', fillcolor='lightgray')
    
    # Connections for Layer 1
    dot.edge('input', 'l1_input_replicate')
    for gpu in range(8):
        dot.edge('l1_input_replicate', f'l1_qkv_gpu{gpu}')
        dot.edge(f'l1_qkv_gpu{gpu}', f'l1_attn_gpu{gpu}')
        dot.edge('l1_kv_allreduce', f'l1_attn_gpu{gpu}')
    for gpu in range(8):
        dot.edge(f'l1_attn_gpu{gpu}', 'l1_attn_aggregate')
    
    dot.edge('l1_attn_aggregate', 'l1_moe_broadcast')
    dot.edge('l1_moe_broadcast', 'l1_gate', style='dashed')
    for gpu in range(8):
        dot.edge('l1_moe_broadcast', f'l1_expert_gpu{gpu}')
        dot.edge('l1_gate', f'l1_expert_gpu{gpu}', style='dashed')
    for gpu in range(8):
        dot.edge(f'l1_expert_gpu{gpu}', 'l1_moe_aggregate')
    
    dot.edge('input', 'l1_residual')
    dot.edge('l1_moe_aggregate', 'l1_residual')
    
    # Connections for Layer 2
    dot.edge('l1_residual', 'l2_input_replicate')
    for gpu in range(8):
        dot.edge('l2_input_replicate', f'l2_qkv_gpu{gpu}')
        dot.edge(f'l2_qkv_gpu{gpu}', f'l2_attn_gpu{gpu}')
        dot.edge('l2_kv_allreduce', f'l2_attn_gpu{gpu}')
    for gpu in range(8):
        dot.edge(f'l2_attn_gpu{gpu}', 'l2_attn_aggregate')
    
    dot.edge('l2_attn_aggregate', 'l2_moe_broadcast')
    dot.edge('l2_moe_broadcast', 'l2_gate', style='dashed')
    for gpu in range(8):
        dot.edge('l2_moe_broadcast', f'l2_expert_gpu{gpu}')
        dot.edge('l2_gate', f'l2_expert_gpu{gpu}', style='dashed')
    for gpu in range(8):
        dot.edge(f'l2_expert_gpu{gpu}', 'l2_moe_aggregate')
    
    dot.edge('l1_residual', 'l2_residual')
    dot.edge('l2_moe_aggregate', 'l2_residual')
    
    # Connections for Layer 3
    dot.edge('l2_residual', 'l3_input_replicate')
    for gpu in range(8):
        dot.edge('l3_input_replicate', f'l3_qkv_gpu{gpu}')
        dot.edge(f'l3_qkv_gpu{gpu}', f'l3_attn_gpu{gpu}')
        dot.edge('l3_kv_allreduce', f'l3_attn_gpu{gpu}')
    for gpu in range(8):
        dot.edge(f'l3_attn_gpu{gpu}', 'l3_attn_aggregate')
    
    dot.edge('l3_attn_aggregate', 'l3_moe_broadcast')
    dot.edge('l3_moe_broadcast', 'l3_gate', style='dashed')
    for gpu in range(8):
        dot.edge('l3_moe_broadcast', f'l3_expert_gpu{gpu}')
        dot.edge('l3_gate', f'l3_expert_gpu{gpu}', style='dashed')
    for gpu in range(8):
        dot.edge(f'l3_expert_gpu{gpu}', 'l3_moe_aggregate')
    
    dot.edge('l2_residual', 'l3_residual')
    dot.edge('l3_moe_aggregate', 'l3_residual')
    
    # Connections for Layer 4
    dot.edge('l3_residual', 'l4_input_replicate')
    for gpu in range(8):
        dot.edge('l4_input_replicate', f'l4_qkv_gpu{gpu}')
        dot.edge(f'l4_qkv_gpu{gpu}', f'l4_attn_gpu{gpu}')
        dot.edge('l4_kv_allreduce', f'l4_attn_gpu{gpu}')
    for gpu in range(8):
        dot.edge(f'l4_attn_gpu{gpu}', 'l4_attn_aggregate')
    
    dot.edge('l4_attn_aggregate', 'l4_moe_broadcast')
    dot.edge('l4_moe_broadcast', 'l4_gate', style='dashed')
    for gpu in range(8):
        dot.edge('l4_moe_broadcast', f'l4_expert_gpu{gpu}')
        dot.edge('l4_gate', f'l4_expert_gpu{gpu}', style='dashed')
    for gpu in range(8):
        dot.edge(f'l4_expert_gpu{gpu}', 'l4_moe_aggregate')
    
    dot.edge('l3_residual', 'l4_residual')
    dot.edge('l4_moe_aggregate', 'l4_residual')
    dot.edge('l4_residual', 'output')
    
    return dot

if __name__ == '__main__':
    dag = create_ma_separation_dag()
    dag.render('/home/wzc/data/file-share/logs/2025-10-09-16-31-25/ma_separation_dag', format='svg', cleanup=False)
    dag.save('/home/wzc/data/file-share/logs/2025-10-09-16-31-25/ma_separation_dag.dot')