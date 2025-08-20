import graphviz

# Create the proposed cross-node expert parallelism DAG
dot = graphviz.Digraph('Proposed_Cross_Node_Expert_Parallelism', format='svg')
dot.attr(rankdir='TB', size='50,50')

# Define styles
with dot.subgraph(name='cluster_legend') as legend:
    legend.attr(label='Legend', style='dashed')
    legend.node('comp', 'Computation', shape='rectangle')
    legend.node('comm', 'Communication', shape='ellipse')
    legend.node('route', 'Routing/Aggregation', shape='parallelogram')

# Global input
dot.node('global_input', 'Global Input\n[1024 tokens, hidden_size]', shape='ellipse', style='filled', fillcolor='lightblue')

# Layer 1
with dot.subgraph(name='cluster_layer1') as layer1:
    layer1.attr(label='Layer 1', style='rounded')
    
    # MHA across all GPUs (tensor parallel)
    layer1.node('l1_mha_all', 'Multi-Head Attention\n[1024×8192]\nAll GPUs', shape='rectangle', style='filled', fillcolor='lightgreen')
    layer1.node('l1_mha_res', 'Residual Add\n[1024×8192]\nAll GPUs', shape='parallelogram', style='filled', fillcolor='yellow')
    
    # Gate for expert selection
    layer1.node('l1_gate', 'Gating Network\n[1024×64 experts]\nAll GPUs', shape='parallelogram', style='filled', fillcolor='orange')
    
    # Create 64 expert nodes, each on a different GPU
    for gpu_id in range(64):
        expert_id = gpu_id
        layer1.node(f'l1_expert_{expert_id}', f'Expert {expert_id}\nMLP [1024×32768]\nGPU {gpu_id}', shape='rectangle', style='filled', fillcolor='lightcoral')
        layer1.node(f'l1_expert_out_{expert_id}', f'Expert {expert_id} Output\n[1024×8192]\nGPU {gpu_id}', shape='ellipse', style='filled', fillcolor='lightyellow')
    
    # Aggregation after experts
    layer1.node('l1_aggregate', 'Aggregate Expert Outputs\n[1024×8192]\nAll GPUs', shape='parallelogram', style='filled', fillcolor='lightpink')
    layer1.node('l1_ffn_res', 'Residual Add\n[1024×8192]\nAll GPUs', shape='parallelogram', style='filled', fillcolor='yellow')

# Layer 2
with dot.subgraph(name='cluster_layer2') as layer2:
    layer2.attr(label='Layer 2', style='rounded')
    
    layer2.node('l2_mha_all', 'Multi-Head Attention\n[1024×8192]\nAll GPUs', shape='rectangle', style='filled', fillcolor='lightgreen')
    layer2.node('l2_mha_res', 'Residual Add\n[1024×8192]\nAll GPUs', shape='parallelogram', style='filled', fillcolor='yellow')
    
    layer2.node('l2_gate', 'Gating Network\n[1024×64 experts]\nAll GPUs', shape='parallelogram', style='filled', fillcolor='orange')
    
    for gpu_id in range(64):
        expert_id = gpu_id
        layer2.node(f'l2_expert_{expert_id}', f'Expert {expert_id}\nMLP [1024×32768]\nGPU {gpu_id}', shape='rectangle', style='filled', fillcolor='lightcoral')
        layer2.node(f'l2_expert_out_{expert_id}', f'Expert {expert_id} Output\n[1024×8192]\nGPU {gpu_id}', shape='ellipse', style='filled', fillcolor='lightyellow')
    
    layer2.node('l2_aggregate', 'Aggregate Expert Outputs\n[1024×8192]\nAll GPUs', shape='parallelogram', style='filled', fillcolor='lightpink')
    layer2.node('l2_ffn_res', 'Residual Add\n[1024×8192]\nAll GPUs', shape='parallelogram', style='filled', fillcolor='yellow')

# Layer 3
with dot.subgraph(name='cluster_layer3') as layer3:
    layer3.attr(label='Layer 3', style='rounded')
    
    layer3.node('l3_mha_all', 'Multi-Head Attention\n[1024×8192]\nAll GPUs', shape='rectangle', style='filled', fillcolor='lightgreen')
    layer3.node('l3_mha_res', 'Residual Add\n[1024×8192]\nAll GPUs', shape='parallelogram', style='filled', fillcolor='yellow')
    
    layer3.node('l3_gate', 'Gating Network\n[1024×64 experts]\nAll GPUs', shape='parallelogram', style='filled', fillcolor='orange')
    
    for gpu_id in range(64):
        expert_id = gpu_id
        layer3.node(f'l3_expert_{expert_id}', f'Expert {expert_id}\nMLP [1024×32768]\nGPU {gpu_id}', shape='rectangle', style='filled', fillcolor='lightcoral')
        layer3.node(f'l3_expert_out_{expert_id}', f'Expert {expert_id} Output\n[1024×8192]\nGPU {gpu_id}', shape='ellipse', style='filled', fillcolor='lightyellow')
    
    layer3.node('l3_aggregate', 'Aggregate Expert Outputs\n[1024×8192]\nAll GPUs', shape='parallelogram', style='filled', fillcolor='lightpink')
    layer3.node('l3_ffn_res', 'Residual Add\n[1024×8192]\nAll GPUs', shape='parallelogram', style='filled', fillcolor='yellow')

# Layer 4
with dot.subgraph(name='cluster_layer4') as layer4:
    layer4.attr(label='Layer 4', style='rounded')
    
    layer4.node('l4_mha_all', 'Multi-Head Attention\n[1024×8192]\nAll GPUs', shape='rectangle', style='filled', fillcolor='lightgreen')
    layer4.node('l4_mha_res', 'Residual Add\n[1024×8192]\nAll GPUs', shape='parallelogram', style='filled', fillcolor='yellow')
    
    layer4.node('l4_gate', 'Gating Network\n[1024×64 experts]\nAll GPUs', shape='parallelogram', style='filled', fillcolor='orange')
    
    for gpu_id in range(64):
        expert_id = gpu_id
        layer4.node(f'l4_expert_{expert_id}', f'Expert {expert_id}\nMLP [1024×32768]\nGPU {gpu_id}', shape='rectangle', style='filled', fillcolor='lightcoral')
        layer4.node(f'l4_expert_out_{expert_id}', f'Expert {expert_id} Output\n[1024×8192]\nGPU {gpu_id}', shape='ellipse', style='filled', fillcolor='lightyellow')
    
    layer4.node('l4_aggregate', 'Aggregate Expert Outputs\n[1024×8192]\nAll GPUs', shape='parallelogram', style='filled', fillcolor='lightpink')
    layer4.node('l4_ffn_res', 'Residual Add\n[1024×8192]\nAll GPUs', shape='parallelogram', style='filled', fillcolor='yellow')

# Global output
dot.node('global_output', 'Global Output\n[1024 tokens, hidden_size]', shape='ellipse', style='filled', fillcolor='lightblue')

# Connect the flow
# Global input to Layer 1 MHA
dot.edge('global_input', 'l1_mha_all')
dot.edge('l1_mha_all', 'l1_mha_res')
dot.edge('global_input', 'l1_mha_res', style='dashed')  # Residual connection

# MHA to Gate
dot.edge('l1_mha_res', 'l1_gate')

# Gate to experts with routing (dashed)
for gpu_id in range(64):
    expert_id = gpu_id
    dot.edge('l1_gate', f'l1_expert_{expert_id}', style='dashed', label=f'Route tokens\nto GPU {gpu_id}')
    dot.edge(f'l1_expert_{expert_id}', f'l1_expert_out_{expert_id}')
    dot.edge(f'l1_expert_out_{expert_id}', 'l1_aggregate')

# Aggregation to residual
dot.edge('l1_aggregate', 'l1_ffn_res')
dot.edge('l1_mha_res', 'l1_ffn_res', style='dashed')  # Residual connection

# Layer 1 to Layer 2
dot.edge('l1_ffn_res', 'l2_mha_all')
dot.edge('l2_mha_all', 'l2_mha_res')
dot.edge('l1_ffn_res', 'l2_mha_res', style='dashed')  # Residual connection

dot.edge('l2_mha_res', 'l2_gate')

for gpu_id in range(64):
    expert_id = gpu_id
    dot.edge('l2_gate', f'l2_expert_{expert_id}', style='dashed', label=f'Route tokens\nto GPU {gpu_id}')
    dot.edge(f'l2_expert_{expert_id}', f'l2_expert_out_{expert_id}')
    dot.edge(f'l2_expert_out_{expert_id}', 'l2_aggregate')

dot.edge('l2_aggregate', 'l2_ffn_res')
dot.edge('l2_mha_res', 'l2_ffn_res', style='dashed')  # Residual connection

# Layer 2 to Layer 3
dot.edge('l2_ffn_res', 'l3_mha_all')
dot.edge('l3_mha_all', 'l3_mha_res')
dot.edge('l2_ffn_res', 'l3_mha_res', style='dashed')  # Residual connection

dot.edge('l3_mha_res', 'l3_gate')

for gpu_id in range(64):
    expert_id = gpu_id
    dot.edge('l3_gate', f'l3_expert_{expert_id}', style='dashed', label=f'Route tokens\nto GPU {gpu_id}')
    dot.edge(f'l3_expert_{expert_id}', f'l3_expert_out_{expert_id}')
    dot.edge(f'l3_expert_out_{expert_id}', 'l3_aggregate')

dot.edge('l3_aggregate', 'l3_ffn_res')
dot.edge('l3_mha_res', 'l3_ffn_res', style='dashed')  # Residual connection

# Layer 3 to Layer 4
dot.edge('l3_ffn_res', 'l4_mha_all')
dot.edge('l4_mha_all', 'l4_mha_res')
dot.edge('l3_ffn_res', 'l4_mha_res', style='dashed')  # Residual connection

dot.edge('l4_mha_res', 'l4_gate')

for gpu_id in range(64):
    expert_id = gpu_id
    dot.edge('l4_gate', f'l4_expert_{expert_id}', style='dashed', label=f'Route tokens\nto GPU {gpu_id}')
    dot.edge(f'l4_expert_{expert_id}', f'l4_expert_out_{expert_id}')
    dot.edge(f'l4_expert_out_{expert_id}', 'l4_aggregate')

dot.edge('l4_aggregate', 'l4_ffn_res')
dot.edge('l4_mha_res', 'l4_ffn_res', style='dashed')  # Residual connection

# Final output
dot.edge('l4_ffn_res', 'global_output')

# Save the DAG
dot.render('/home/wzc/data/file-share/submission/proposed_cross_node_expert_parallelism', format='svg', cleanup=False)
print("Proposed DAG saved to: /home/wzc/data/file-share/submission/proposed_cross_node_expert_parallelism.svg")