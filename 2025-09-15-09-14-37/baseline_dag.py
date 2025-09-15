#!/usr/bin/env python3

import graphviz

# Create a new directed graph for baseline deployment
baseline = graphviz.Digraph('baseline_moe_deployment', comment='Baseline MoE Deployment (TP=8, PP=2)')
baseline.attr(rankdir='TB', size='20,30')
baseline.attr('node', shape='rectangle', style='filled', fillcolor='lightblue')

# Define node attributes with dimensions
# Model dimensions
batch_size = 1024
seq_len = 10000
token_dim = 8192
num_heads = 16
head_dim = 512
hidden_size = 32768

# Baseline uses 16 GPUs total, 2 pipeline stages, 8 GPUs per stage
# Each GPU has 4 experts

# Input node
baseline.node('input', f'Total Input\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: all GPUs', 
              shape='ellipse', fillcolor='lightgreen')

# Pipeline Stage 1 (GPUs 0-7)
baseline.attr('node', fillcolor='lightcoral')
baseline.node('stage1_mha', f'Pipeline Stage 1 MHA\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: 0-7', 
              shape='rectangle')

# Expert computation for stage 1 - 4 experts per GPU
for gpu_id in range(8):
    for expert_id in range(4):
        expert_num = gpu_id * 4 + expert_id
        baseline.node(f'stage1_expert_{gpu_id}_{expert_id}', 
                      f'Expert {expert_num}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim//8}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim//8}]\\nGPU: {gpu_id}',
                      shape='rectangle', fillcolor='lightyellow')

# Gate for stage 1
baseline.node('stage1_gate', f'Gate Selection\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: routing decisions\\nGPU: 0-7',
              shape='parallelogram', fillcolor='lightpink')

# Residual add for stage 1
baseline.node('stage1_residual', f'Residual Add\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0-7',
              shape='rectangle', fillcolor='lightgreen')

# Communication between stages
baseline.node('comm_stage1_to_stage2', f'Cross-Stage Communication\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 0-7 → 8-15',
              shape='ellipse', fillcolor='lightgray')

# Pipeline Stage 2 (GPUs 8-15)
baseline.attr('node', fillcolor='lightcoral')
baseline.node('stage2_mha', f'Pipeline Stage 2 MHA\\nInput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, heads={num_heads}, d_k={head_dim}]\\nGPU: 8-15',
              shape='rectangle')

# Expert computation for stage 2 - 4 experts per GPU
for gpu_id in range(8, 16):
    for expert_id in range(4):
        expert_num = (gpu_id - 8) * 4 + expert_id
        baseline.node(f'stage2_expert_{gpu_id}_{expert_id}', 
                      f'Expert {expert_num + 32}\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim//8}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim//8}]\\nGPU: {gpu_id}',
                      shape='rectangle', fillcolor='lightyellow')

# Gate for stage 2
baseline.node('stage2_gate', f'Gate Selection\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: routing decisions\\nGPU: 8-15',
              shape='parallelogram', fillcolor='lightpink')

# Residual add for stage 2
baseline.node('stage2_residual', f'Residual Add\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nOutput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: 8-15',
              shape='rectangle', fillcolor='lightgreen')

# Output
baseline.node('output', f'Total Output\\nInput: [batch_size={batch_size}, seq_len={seq_len}, token_dim={token_dim}]\\nGPU: all GPUs',
              shape='ellipse', fillcolor='lightgreen')

# Connect nodes
baseline.edge('input', 'stage1_mha')
baseline.edge('stage1_mha', 'stage1_gate')
baseline.edge('stage1_gate', 'stage1_expert_0_0', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_0_1', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_0_2', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_0_3', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_1_0', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_1_1', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_1_2', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_1_3', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_2_0', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_2_1', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_2_2', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_2_3', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_3_0', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_3_1', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_3_2', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_3_3', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_4_0', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_4_1', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_4_2', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_4_3', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_5_0', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_5_1', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_5_2', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_5_3', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_6_0', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_6_1', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_6_2', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_6_3', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_7_0', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_7_1', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_7_2', style='dashed')
baseline.edge('stage1_gate', 'stage1_expert_7_3', style='dashed')

# Connect experts to residual
for gpu_id in range(8):
    for expert_id in range(4):
        baseline.edge(f'stage1_expert_{gpu_id}_{expert_id}', 'stage1_residual')

baseline.edge('stage1_residual', 'comm_stage1_to_stage2')
baseline.edge('comm_stage1_to_stage2', 'stage2_mha')
baseline.edge('stage2_mha', 'stage2_gate')
baseline.edge('stage2_gate', 'stage2_expert_8_0', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_8_1', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_8_2', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_8_3', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_9_0', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_9_1', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_9_2', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_9_3', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_10_0', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_10_1', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_10_2', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_10_3', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_11_0', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_11_1', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_11_2', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_11_3', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_12_0', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_12_1', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_12_2', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_12_3', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_13_0', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_13_1', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_13_2', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_13_3', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_14_0', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_14_1', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_14_2', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_14_3', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_15_0', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_15_1', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_15_2', style='dashed')
baseline.edge('stage2_gate', 'stage2_expert_15_3', style='dashed')

# Connect stage 2 experts to residual
for gpu_id in range(8, 16):
    for expert_id in range(4):
        baseline.edge(f'stage2_expert_{gpu_id}_{expert_id}', 'stage2_residual')

baseline.edge('stage2_residual', 'output')

# Save the baseline DAG
baseline.render('/home/wzc/data/file-share/2025-09-15-09-14-37/baseline_dag', format='svg', cleanup=False)
baseline.save('/home/wzc/data/file-share/2025-09-15-09-14-37/baseline_dag.dot')

print("Baseline DAG generated successfully!")