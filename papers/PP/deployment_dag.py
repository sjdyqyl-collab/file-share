#!/usr/bin/env python3
"""
Generate a complete deployment DAG for the layer-wise partitioning strategy
across 16 GPUs as described in the paper.
"""

import graphviz

# Create a new directed graph
dot = graphviz.Digraph('LayerWiseDeployment', 
                       comment='Layer-wise Model Deployment across 16 GPUs',
                       graph_attr={
                           'rankdir': 'TB',
                           'splines': 'ortho',
                           'nodesep': '0.5',
                           'ranksep': '1.0',
                           'bgcolor': 'white'
                       })

# Define GPU clusters
gpu_clusters = {}
for gpu_id in range(16):
    with dot.subgraph(name=f'cluster_gpu_{gpu_id}') as c:
        c.attr(label=f'GPU {gpu_id}', style='rounded, filled', color='lightblue', fillcolor='lightblue')
        gpu_clusters[gpu_id] = c

# Model dimensions
batch_size = 1024
seq_len = 2048  # Typical sequence length
hidden_dim = 4096  # Typical hidden dimension
num_heads = 32
head_dim = hidden_dim // num_heads
expert_dim = 4 * hidden_dim  # FFN expansion factor
num_experts = 8  # For MoE model

# Define colors for different components
colors = {
    'input': 'lightgreen',
    'attention': 'lightcoral',
    'dense_ffn': 'lightyellow',
    'moe_expert': 'lightpink',
    'moe_gate': 'lightsalmon',
    'output': 'lightsteelblue',
    'comm': 'orange'
}

# Add input node
dot.node('input', f'Input\nBatch: {batch_size}x{seq_len}x{hidden_dim}\nFP16', 
         shape='ellipse', style='filled', fillcolor=colors['input'])

# Layer-wise partitioning across 16 GPUs
# Since we have 4 layers and 16 GPUs, we'll distribute each layer across 4 GPUs
# This creates 4 partitions, each with 4 GPUs handling different parts of a layer

layers = ['Layer1', 'Layer2', 'Layer3', 'Layer4']
partition_size = 4  # 4 GPUs per layer

# Track nodes for connections
prev_nodes = ['input']

for layer_idx, layer_name in enumerate(layers):
    # Calculate GPU range for this layer
    start_gpu = layer_idx * partition_size
    end_gpu = start_gpu + partition_size
    
    # Dense model components
    # Each layer has: LayerNorm -> Multi-Head Attention -> LayerNorm -> FFN/Experts
    
    # Multi-Head Attention partitioning (Column-wise on QKV, Row-wise on Output)
    # Split attention heads across 4 GPUs
    heads_per_gpu = num_heads // partition_size
    
    # Create attention nodes for this layer
    attn_nodes = []
    for i, gpu_id in enumerate(range(start_gpu, end_gpu)):
        gpu_clusters[gpu_id].node(f'{layer_name}_attn_qkv_{i}', 
                                 f'QKV Linear\nCol Parallel\nGPU {gpu_id}\n{heads_per_gpu}x{head_dim} heads\nFP16',
                                 shape='box', style='filled', fillcolor=colors['attention'])
        
        gpu_clusters[gpu_id].node(f'{layer_name}_attn_compute_{i}', 
                                 f'Attention Compute\nGPU {gpu_id}\n{heads_per_gpu}x{head_dim} heads\nSoftmax & MatMul',
                                 shape='box', style='filled', fillcolor=colors['attention'])
        
        gpu_clusters[gpu_id].node(f'{layer_name}_attn_out_{i}', 
                                 f'Output Linear\nRow Parallel\nGPU {gpu_id}\n{hidden_dim} dim',
                                 shape='box', style='filled', fillcolor=colors['attention'])
        
        attn_nodes.extend([f'{layer_name}_attn_qkv_{i}', f'{layer_name}_attn_compute_{i}', f'{layer_name}_attn_out_{i}'])
    
    # Dense FFN components (for dense model variant)
    ffn_nodes = []
    for i, gpu_id in enumerate(range(start_gpu, end_gpu)):
        # Split FFN across GPUs (Column-wise on first linear, Row-wise on second)
        ffn_inner_dim = expert_dim // partition_size
        
        gpu_clusters[gpu_id].node(f'{layer_name}_ffn_gate_{i}', 
                                 f'FFN Gate\nCol Parallel\nGPU {gpu_id}\n{ffn_inner_dim}x{hidden_dim}',
                                 shape='box', style='filled', fillcolor=colors['dense_ffn'])
        
        gpu_clusters[gpu_id].node(f'{layer_name}_ffn_up_{i}', 
                                 f'FFN Up\nCol Parallel\nGPU {gpu_id}\n{ffn_inner_dim}x{hidden_dim}',
                                 shape='box', style='filled', fillcolor=colors['dense_ffn'])
        
        gpu_clusters[gpu_id].node(f'{layer_name}_ffn_down_{i}', 
                                 f'FFN Down\nRow Parallel\nGPU {gpu_id}\n{hidden_dim}x{ffn_inner_dim}',
                                 shape='box', style='filled', fillcolor=colors['dense_ffn'])
        
        ffn_nodes.extend([f'{layer_name}_ffn_gate_{i}', f'{layer_name}_ffn_up_{i}', f'{layer_name}_ffn_down_{i}'])
    
    # MoE components (for MoE model variant)
    moe_nodes = []
    for i, gpu_id in enumerate(range(start_gpu, end_gpu)):
        # Each GPU handles 2 experts (8 experts total / 4 GPUs)
        experts_per_gpu = num_experts // partition_size
        
        gpu_clusters[gpu_id].node(f'{layer_name}_moe_gate_{i}', 
                                 f'MoE Gate\nGPU {gpu_id}\nTop-2 routing',
                                 shape='diamond', style='filled', fillcolor=colors['moe_gate'])
        
        for expert_idx in range(experts_per_gpu):
            expert_id = i * experts_per_gpu + expert_idx
            gpu_clusters[gpu_id].node(f'{layer_name}_expert_{expert_id}', 
                                     f'Expert {expert_id}\nGPU {gpu_id}\n{expert_dim}x{hidden_dim}',
                                     shape='box', style='filled', fillcolor=colors['moe_expert'])
            
            moe_nodes.append(f'{layer_name}_expert_{expert_id}')
        
        moe_nodes.append(f'{layer_name}_moe_gate_{i}')
    
    # Communication nodes for AllGather/Reduce
    comm_nodes = []
    for i, gpu_id in enumerate(range(start_gpu, end_gpu)):
        gpu_clusters[gpu_id].node(f'{layer_name}_allgather_{i}', 
                                 f'AllGather\nGPU {gpu_id}',
                                 shape='circle', style='filled', fillcolor=colors['comm'])
        
        gpu_clusters[gpu_id].node(f'{layer_name}_reduce_{i}', 
                                 f'Reduce\nGPU {gpu_id}',
                                 shape='circle', style='filled', fillcolor=colors['comm'])
        
        comm_nodes.extend([f'{layer_name}_allgather_{i}', f'{layer_name}_reduce_{i}'])
    
    # Connect previous layer to current layer
    for prev_node in prev_nodes:
        for i in range(partition_size):
            dot.edge(prev_node, f'{layer_name}_attn_qkv_{i}')
    
    # Connect attention components
    for i in range(partition_size):
        dot.edge(f'{layer_name}_attn_qkv_{i}', f'{layer_name}_attn_compute_{i}')
        dot.edge(f'{layer_name}_attn_compute_{i}', f'{layer_name}_attn_out_{i}')
        dot.edge(f'{layer_name}_attn_out_{i}', f'{layer_name}_allgather_{i}')
        dot.edge(f'{layer_name}_allgather_{i}', f'{layer_name}_reduce_{i}')
    
    # Connect to FFN/MoE based on model type
    for i in range(partition_size):
        # For dense model - connect to FFN
        dot.edge(f'{layer_name}_reduce_{i}', f'{layer_name}_ffn_gate_{i}')
        dot.edge(f'{layer_name}_ffn_gate_{i}', f'{layer_name}_ffn_up_{i}')
        dot.edge(f'{layer_name}_ffn_up_{i}', f'{layer_name}_ffn_down_{i}')
        
        # For MoE model - connect to MoE gate then experts
        dot.edge(f'{layer_name}_reduce_{i}', f'{layer_name}_moe_gate_{i}')
        
        experts_per_gpu = num_experts // partition_size
        for expert_idx in range(experts_per_gpu):
            expert_id = i * experts_per_gpu + expert_idx
            dot.edge(f'{layer_name}_moe_gate_{i}', f'{layer_name}_expert_{expert_id}')
    
    # Update prev_nodes for next layer
    prev_nodes = [f'{layer_name}_ffn_down_{i}' for i in range(partition_size)] + \
                 [f'{layer_name}_expert_{i*2+expert_idx}' for i in range(partition_size) for expert_idx in range(2)]

# Add final output node
dot.node('output', f'Output\nBatch: {batch_size}x{seq_len}x{hidden_dim}\nFP16', 
         shape='ellipse', style='filled', fillcolor=colors['output'])

# Connect final layer to output
for prev_node in prev_nodes:
    if 'ffn_down' in prev_node or 'expert_' in prev_node:
        dot.edge(prev_node, 'output')

# Add communication edges between GPUs for tensor parallelism
for layer_idx in range(4):
    start_gpu = layer_idx * partition_size
    
    # AllGather communication after attention output
    for i in range(partition_size):
        for j in range(partition_size):
            if i != j:
                from_gpu = start_gpu + i
                to_gpu = start_gpu + j
                dot.edge(f'Layer{layer_idx+1}_attn_out_{i}', 
                        f'Layer{layer_idx+1}_allgather_{j}',
                        color='red', style='dashed', constraint='false')
    
    # Reduce communication after FFN/MoE
    for i in range(partition_size):
        for j in range(partition_size):
            if i != j:
                from_gpu = start_gpu + i
                to_gpu = start_gpu + j
                dot.edge(f'Layer{layer_idx+1}_ffn_down_{i}' if layer_idx < 3 else 'output',
                        f'Layer{layer_idx+1}_reduce_{j}',
                        color='blue', style='dashed', constraint='false')

# Save the graph
dot.format = 'dot'
dot.render('/home/wzc/data/papers/PP/layer_wise_deployment', cleanup=False)

# Also generate the raw DOT content
dot_content = dot.source
with open('/home/wzc/data/papers/PP/layer_wise_deployment.dot', 'w') as f:
    f.write(dot_content)

print("Generated deployment DAG files:")
print("- layer_wise_deployment.dot (Graphviz source)")
print("- layer_wise_deployment.pdf (rendered diagram)")
print("\nDOT content:")
print(dot_content)