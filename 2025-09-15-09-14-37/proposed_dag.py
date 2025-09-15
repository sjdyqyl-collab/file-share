#!/usr/bin/env python3

import graphviz

# Create a new directed graph for proposed cross-node expert parallelism
proposed = graphviz.Digraph('proposed_moe_deployment', comment='Proposed Cross-Node Expert Parallelism')
proposed.attr(rankdir='TB', size='30,40')
proposed.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')

# Define node attributes with dimensions
batch_size = 1024
seq_len = 10000
token_dim = 8192
num_heads = 16
head_dim = 512
hidden_size = 32768

# Proposed uses 64 GPUs total, 1 expert per GPU
# 4 layers × 16 experts per layer = 64 experts total

# Input node
proposed.node('input', f'Total Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: all GPUs', 
              shape='ellipse', fillcolor='lightgreen')

# Layer 1
proposed.attr('node', fillcolor='lightcoral')
proposed.node('layer1_mha', f'Layer 1 MHA\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: all GPUs',
              shape='rectangle')

# Gate for layer 1
proposed.node('layer1_gate', f'Layer 1 Gate Selection\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: routing decisions\\nGPU: all GPUs',
              shape='parallelogram', fillcolor='lightpink')

# Experts for layer 1 - 1 expert per GPU, GPUs 0-15
for gpu_id in range(16):
    expert_id = gpu_id
    proposed.node(f'layer1_expert_{gpu_id}', 
                  f'Layer 1 Expert {expert_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: {gpu_id}',
                  shape='rectangle', fillcolor='lightyellow')

# Token routing and aggregation for layer 1
proposed.node('layer1_token_routing', f'Layer 1 Token Routing\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: distributed tokens\\nGPU: all GPUs',
              shape='ellipse', fillcolor='lightgray')

proposed.node('layer1_token_aggregation', f'Layer 1 Token Aggregation\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: all GPUs',
              shape='ellipse', fillcolor='lightgray')

# Residual add for layer 1
proposed.node('layer1_residual', f'Layer 1 Residual Add\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: all GPUs',
              shape='rectangle', fillcolor='lightgreen')

# Layer 2
proposed.attr('node', fillcolor='lightcoral')
proposed.node('layer2_mha', f'Layer 2 MHA\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: all GPUs',
              shape='rectangle')

# Gate for layer 2
proposed.node('layer2_gate', f'Layer 2 Gate Selection\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: routing decisions\\nGPU: all GPUs',
              shape='parallelogram', fillcolor='lightpink')

# Experts for layer 2 - 1 expert per GPU, GPUs 16-31
for gpu_id in range(16, 32):
    expert_id = gpu_id - 16
    proposed.node(f'layer2_expert_{gpu_id}', 
                  f'Layer 2 Expert {expert_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: {gpu_id}',
                  shape='rectangle', fillcolor='lightyellow')

# Token routing and aggregation for layer 2
proposed.node('layer2_token_routing', f'Layer 2 Token Routing\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: distributed tokens\\nGPU: all GPUs',
              shape='ellipse', fillcolor='lightgray')

proposed.node('layer2_token_aggregation', f'Layer 2 Token Aggregation\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: all GPUs',
              shape='ellipse', fillcolor='lightgray')

# Residual add for layer 2
proposed.node('layer2_residual', f'Layer 2 Residual Add\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: all GPUs',
              shape='rectangle', fillcolor='lightgreen')

# Layer 3
proposed.attr('node', fillcolor='lightcoral')
proposed.node('layer3_mha', f'Layer 3 MHA\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: all GPUs',
              shape='rectangle')

# Gate for layer 3
proposed.node('layer3_gate', f'Layer 3 Gate Selection\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: routing decisions\\nGPU: all GPUs',
              shape='parallelogram', fillcolor='lightpink')

# Experts for layer 3 - 1 expert per GPU, GPUs 32-47
for gpu_id in range(32, 48):
    expert_id = gpu_id - 32
    proposed.node(f'layer3_expert_{gpu_id}', 
                  f'Layer 3 Expert {expert_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: {gpu_id}',
                  shape='rectangle', fillcolor='lightyellow')

# Token routing and aggregation for layer 3
proposed.node('layer3_token_routing', f'Layer 3 Token Routing\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: distributed tokens\\nGPU: all GPUs',
              shape='ellipse', fillcolor='lightgray')

proposed.node('layer3_token_aggregation', f'Layer 3 Token Aggregation\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: all GPUs',
              shape='ellipse', fillcolor='lightgray')

# Residual add for layer 3
proposed.node('layer3_residual', f'Layer 3 Residual Add\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: all GPUs',
              shape='rectangle', fillcolor='lightgreen')

# Layer 4
proposed.attr('node', fillcolor='lightcoral')
proposed.node('layer4_mha', f'Layer 4 MHA\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: all GPUs',
              shape='rectangle')

# Gate for layer 4
proposed.node('layer4_gate', f'Layer 4 Gate Selection\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: routing decisions\\nGPU: all GPUs',
              shape='parallelogram', fillcolor='lightpink')

# Experts for layer 4 - 1 expert per GPU, GPUs 48-63
for gpu_id in range(48, 64):
    expert_id = gpu_id - 48
    proposed.node(f'layer4_expert_{gpu_id}', 
                  f'Layer 4 Expert {expert_id}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: {gpu_id}',
                  shape='rectangle', fillcolor='lightyellow')

# Token routing and aggregation for layer 4
proposed.node('layer4_token_routing', f'Layer 4 Token Routing\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: distributed tokens\\nGPU: all GPUs',
              shape='ellipse', fillcolor='lightgray')

proposed.node('layer4_token_aggregation', f'Layer 4 Token Aggregation\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: all GPUs',
              shape='ellipse', fillcolor='lightgray')

# Residual add for layer 4
proposed.node('layer4_residual', f'Layer 4 Residual Add\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: all GPUs',
              shape='rectangle', fillcolor='lightgreen')

# Output
proposed.node('output', f'Total Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: all GPUs',
              shape='ellipse', fillcolor='lightgreen')

# Connect nodes - Layer 1
proposed.edge('input', 'layer1_mha')
proposed.edge('layer1_mha', 'layer1_gate')
for gpu_id in range(16):
    proposed.edge('layer1_gate', f'layer1_expert_{gpu_id}', style='dashed')
    proposed.edge('layer1_token_routing', f'layer1_expert_{gpu_id}')
    proposed.edge(f'layer1_expert_{gpu_id}', 'layer1_token_aggregation')
proposed.edge('layer1_mha', 'layer1_token_routing')
proposed.edge('layer1_token_aggregation', 'layer1_residual')

# Connect Layer 1 to Layer 2
proposed.edge('layer1_residual', 'layer2_mha')
proposed.edge('layer2_mha', 'layer2_gate')
for gpu_id in range(16, 32):
    proposed.edge('layer2_gate', f'layer2_expert_{gpu_id}', style='dashed')
    proposed.edge('layer2_token_routing', f'layer2_expert_{gpu_id}')
    proposed.edge(f'layer2_expert_{gpu_id}', 'layer2_token_aggregation')
proposed.edge('layer2_mha', 'layer2_token_routing')
proposed.edge('layer2_token_aggregation', 'layer2_residual')

# Connect Layer 2 to Layer 3
proposed.edge('layer2_residual', 'layer3_mha')
proposed.edge('layer3_mha', 'layer3_gate')
for gpu_id in range(32, 48):
    proposed.edge('layer3_gate', f'layer3_expert_{gpu_id}', style='dashed')
    proposed.edge('layer3_token_routing', f'layer3_expert_{gpu_id}')
    proposed.edge(f'layer3_expert_{gpu_id}', 'layer3_token_aggregation')
proposed.edge('layer3_mha', 'layer3_token_routing')
proposed.edge('layer3_token_aggregation', 'layer3_residual')

# Connect Layer 3 to Layer 4
proposed.edge('layer3_residual', 'layer4_mha')
proposed.edge('layer4_mha', 'layer4_gate')
for gpu_id in range(48, 64):
    proposed.edge('layer4_gate', f'layer4_expert_{gpu_id}', style='dashed')
    proposed.edge('layer4_token_routing', f'layer4_expert_{gpu_id}')
    proposed.edge(f'layer4_expert_{gpu_id}', 'layer4_token_aggregation')
proposed.edge('layer4_mha', 'layer4_token_routing')
proposed.edge('layer4_token_aggregation', 'layer4_residual')

# Connect to output
proposed.edge('layer4_residual', 'output')

# Save the proposed DAG
proposed.render('/home/wzc/data/file-share/2025-09-15-09-14-37/proposed_dag', format='svg', cleanup=False)
proposed.save('/home/wzc/data/file-share/2025-09-15-09-14-37/proposed_dag.dot')

print("Proposed DAG generated successfully!")