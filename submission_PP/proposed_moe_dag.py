import graphviz

# Create proposed layer-wise deployment DAG for MoE model
# With 16 GPUs, we distribute 16 layers evenly: 1 layer per GPU
# Each MoE layer has 8 experts, but all experts for a layer are on the same GPU
dot = graphviz.Digraph('proposed_moe', comment='MoE Model - Proposed Layer-wise Cache-Fitting Deployment')
dot.attr(rankdir='TB', size='30,30')

# Define colors for different GPUs
colors = [
    '#ffcccc', '#ccffcc', '#ccccff', '#ffffcc',  # GPUs 0-3
    '#ffccff', '#ccffff', '#ffddcc', '#ddffcc',  # GPUs 4-7
    '#ffaaaa', '#aaffaa', '#aaaaff', '#ffffaa',  # GPUs 8-11
    '#ffaaff', '#aaffff', '#ffbbaa', '#bbffaa'   # GPUs 12-15
]

# Input node
dot.node('input', 'Input\n[1024, seq_len, 8192]\nGPU 0', shape='ellipse', style='filled', fillcolor='lightgray')

# Create layer nodes for each GPU (1 layer per GPU)
for gpu_id in range(16):
    layer = gpu_id + 1
    
    # Multi-Head Attention
    dot.node(f'layer{layer}_mha_q', f'Layer{layer}\nMHA Q Linear\n[1024, seq_len, 8192]->[1024, seq_len, 8192]\nGPU {gpu_id}', 
             fillcolor=colors[gpu_id])
    dot.node(f'layer{layer}_mha_k', f'Layer{layer}\nMHA K Linear\n[1024, seq_len, 8192]->[1024, seq_len, 8192]\nGPU {gpu_id}', 
             fillcolor=colors[gpu_id])
    dot.node(f'layer{layer}_mha_v', f'Layer{layer}\nMHA V Linear\n[1024, seq_len, 8192]->[1024, seq_len, 8192]\nGPU {gpu_id}', 
             fillcolor=colors[gpu_id])
    dot.node(f'layer{layer}_mha_attn', f'Layer{layer}\nMHA Attention\n[1024, seq_len, 8192]->[1024, seq_len, 8192]\nGPU {gpu_id}', 
             fillcolor=colors[gpu_id])
    dot.node(f'layer{layer}_mha_out', f'Layer{layer}\nMHA Output Linear\n[1024, seq_len, 8192]->[1024, seq_len, 8192]\nGPU {gpu_id}', 
             fillcolor=colors[gpu_id])
    dot.node(f'layer{layer}_mha_add', f'Layer{layer}\nMHA Residual Add\n[1024, seq_len, 8192]\nGPU {gpu_id}', 
             shape='parallelogram', fillcolor=colors[gpu_id])
    
    # MoE Gate
    dot.node(f'layer{layer}_gate', f'Layer{layer}\nMoE Gate\n[1024, seq_len, 8192]->[1024, seq_len, 8]\nGPU {gpu_id}', 
             shape='diamond', fillcolor=colors[gpu_id])
    
    # MoE Experts (8 experts, all on the same GPU)
    for expert in range(8):
        dot.node(f'layer{layer}_expert{expert}_ffn1', f'Layer{layer}\nExpert{expert} FFN1\n[1024, seq_len, 8192]->[1024, seq_len, 32768]\nGPU {gpu_id}', 
                 fillcolor=colors[gpu_id])
        dot.node(f'layer{layer}_expert{expert}_ffn2', f'Layer{layer}\nExpert{expert} FFN2\n[1024, seq_len, 32768]->[1024, seq_len, 8192]\nGPU {gpu_id}', 
                 fillcolor=colors[gpu_id])
    
    # MoE Aggregation
    dot.node(f'layer{layer}_moe_agg', f'Layer{layer}\nMoE Aggregation\n[1024, seq_len, 8192]\nGPU {gpu_id}', 
             shape='parallelogram', fillcolor=colors[gpu_id])
    dot.node(f'layer{layer}_moe_add', f'Layer{layer}\nMoE Residual Add\n[1024, seq_len, 8192]\nGPU {gpu_id}', 
             shape='parallelogram', fillcolor=colors[gpu_id])
    
    # LayerNorm nodes
    dot.node(f'layer{layer}_ln1', f'Layer{layer}\nLayerNorm 1\n[1024, seq_len, 8192]\nGPU {gpu_id}', 
             fillcolor=colors[gpu_id])
    dot.node(f'layer{layer}_ln2', f'Layer{layer}\nLayerNorm 2\n[1024, seq_len, 8192]\nGPU {gpu_id}', 
             fillcolor=colors[gpu_id])

# Communication nodes between GPUs (point-to-point)
for gpu_id in range(15):
    dot.node(f'comm_{gpu_id}_{gpu_id+1}', f'GPU{gpu_id} -> GPU{gpu_id+1}\n[1024, seq_len, 8192]\nCache-to-Cache Transfer', 
             shape='ellipse', style='filled', fillcolor='orange')

# Output node
dot.node('output', 'Output\n[1024, seq_len, 8192]\nGPU 15', shape='ellipse', style='filled', fillcolor='lightgray')

# Connect the DAG
# Input to first layer
dot.edge('input', 'layer1_ln1')

for layer in range(1, 17):
    gpu_id = layer - 1
    
    # MHA path
    dot.edge(f'layer{layer}_ln1', f'layer{layer}_mha_q')
    dot.edge(f'layer{layer}_ln1', f'layer{layer}_mha_k')
    dot.edge(f'layer{layer}_ln1', f'layer{layer}_mha_v')
    dot.edge(f'layer{layer}_mha_q', f'layer{layer}_mha_attn')
    dot.edge(f'layer{layer}_mha_k', f'layer{layer}_mha_attn')
    dot.edge(f'layer{layer}_mha_v', f'layer{layer}_mha_attn')
    dot.edge(f'layer{layer}_mha_attn', f'layer{layer}_mha_out')
    dot.edge(f'layer{layer}_mha_out', f'layer{layer}_mha_add')
    
    # Add residual connection
    if layer == 1:
        dot.edge('input', f'layer1_mha_add')
    else:
        dot.edge(f'comm_{gpu_id-1}_{gpu_id}', f'layer{layer}_mha_add')
    
    dot.edge(f'layer{layer}_mha_add', f'layer{layer}_ln2')
    
    # MoE path
    dot.edge(f'layer{layer}_ln2', f'layer{layer}_gate')
    
    # Connect gate to experts (dashed lines for routing)
    for expert in range(8):
        dot.edge(f'layer{layer}_gate', f'layer{layer}_expert{expert}_ffn1', style='dashed')
        dot.edge(f'layer{layer}_ln2', f'layer{layer}_expert{expert}_ffn1', style='dashed')
        dot.edge(f'layer{layer}_expert{expert}_ffn1', f'layer{layer}_expert{expert}_ffn2')
        dot.edge(f'layer{layer}_expert{expert}_ffn2', f'layer{layer}_moe_agg')
    
    dot.edge(f'layer{layer}_moe_agg', f'layer{layer}_moe_add')
    dot.edge(f'layer{layer}_mha_add', f'layer{layer}_moe_add')
    
    # Connect to next layer or output
    if layer < 16:
        dot.edge(f'layer{layer}_moe_add', f'comm_{gpu_id}_{gpu_id+1}')
    else:
        dot.edge(f'layer{layer}_moe_add', 'output')

# Add memory placement annotations
dot.attr(label='Proposed Layer-wise Deployment: 16 MoE layers distributed across 16 GPUs\nEach layer with 8 experts fits entirely in SRAM/L2 cache of assigned GPU')

# Save the DAG
dot.format = 'svg'
dot.render('/home/wzc/data/file-share/submission/proposed_moe_dag', cleanup=True)

print("Proposed MoE DAG generated successfully")